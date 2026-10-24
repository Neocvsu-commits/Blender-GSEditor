"""自动更新：到 GitHub Releases 查最新版，面板顶部提示，可一键覆盖安装。

和其他 Neo 插件同一套做法：注册时后台查一次（不挡启动），主面板顶部显示版本和
检查结果，有新版就多出「一键更新 / 更新说明」。区别是这个插件是扩展
（blender_manifest.toml），没有 bl_info，版本从清单里读。
"""

import json
import os
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile

import bpy

OWNER = "Neocvsu-commits"
REPO = "Blender-GSEditor"

# 安装目录 = 本文件所在目录；一键更新就覆盖到这里
ADDON_DIR = os.path.dirname(os.path.abspath(__file__))

# 清单读不出来时的兜底版本，正常情况下用不到
FALLBACK_VERSION = (1, 1, 0)

USER_AGENT = "Blender-GSEditor-Update-Checker"
API_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/latest"

# 这些目录/文件不进安装目录：缓存、版本库、字节码
_SKIP_DIRS = {"__pycache__", ".git", ".github", ".vs", ".idea"}
_SKIP_SUFFIXES = (".pyc", ".pyo")

_cache = {}
_cache_lock = threading.Lock()
_polling = False


# ---------------------------------------------------------------- 版本号
def _parse_version(text):
    """'v1.2.3' / '1.2' / '1.2.0-beta' → (1, 2, 3)；认不出来返回 None。"""
    if not text:
        return None
    core = str(text).strip().lstrip("vV")
    parts = []
    for piece in core.split(".")[:3]:
        digits = ""
        for char in piece:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


def version_text(version=None):
    version = VERSION if version is None else version
    return ".".join(str(part) for part in version)


def _version_newer(latest, current):
    """latest > current 就返回 True，短的版本号补零比较。"""
    length = max(len(latest), len(current))
    a = list(latest) + [0] * (length - len(latest))
    b = list(current) + [0] * (length - len(current))
    return a > b


def _read_manifest_version():
    """扩展的唯一版本来源：blender_manifest.toml。"""
    path = os.path.join(ADDON_DIR, "blender_manifest.toml")
    try:
        try:
            import tomllib
        except ImportError:                    # 3.11 以前的 Python
            tomllib = None
        if tomllib is not None:
            with open(path, "rb") as handle:
                return _parse_version(tomllib.load(handle).get("version"))
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition("=")
                if key.strip() == "version":
                    return _parse_version(value.strip().strip("\"'"))
    except (OSError, ValueError):
        pass
    return None


VERSION = _read_manifest_version() or FALLBACK_VERSION


# ---------------------------------------------------------------- 查更新
def _store(result):
    with _cache_lock:
        _cache[REPO] = result


def _get_json(url, timeout=10):
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _pick_release_zip(release):
    """优先发布里带的扩展包（gs_editor-*.zip），没有就退到源码归档。"""
    assets = [asset for asset in release.get("assets", [])
              if str(asset.get("name", "")).lower().endswith(".zip")]
    assets.sort(key=lambda asset: not str(asset.get("name", "")).lower().startswith("gs_editor"))
    if assets:
        return assets[0].get("browser_download_url", ""), assets[0].get("name", "")
    tag = release.get("tag_name", "")
    if not tag:
        return "", ""
    return f"https://github.com/{OWNER}/{REPO}/archive/refs/tags/{tag}.zip", ""


def _check_thread():
    _store({"status": "checking", "current_version": version_text()})
    try:
        release = _get_json(API_URL)
        tag = str(release.get("tag_name") or "")
        latest = _parse_version(tag)
        if latest is None:
            _store({"status": "error", "error": f"认不出发布标签：{tag or '空'}"})
            return
        base = {
            "current_version": version_text(),
            "latest_version": version_text(latest),
            "tag_name": tag,
            "html_url": release.get("html_url", ""),
        }
        if not _version_newer(latest, VERSION):
            _store(dict(base, status="no_update"))
            return
        zip_url, zip_name = _pick_release_zip(release)
        if not zip_url:
            _store({"status": "error", "error": "发布里没找到下载包"})
            return
        _store(dict(base, status="has_update", zip_url=zip_url, zip_name=zip_name))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            _store({"status": "no_release", "current_version": version_text()})
        elif exc.code in {403, 429}:
            _store({"status": "error", "error": "GitHub 请求太频繁，等一会儿再试"})
        else:
            _store({"status": "error", "error": f"GitHub API HTTP {exc.code}"})
    except Exception as exc:                   # noqa: BLE001  网络问题都归到这里
        _store({"status": "error", "error": str(exc) or exc.__class__.__name__})


def _redraw():
    """让 3D 视图 / 偏好设置重画一次，状态文字自己更新，不用用户去别处点一下。"""
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type in {'VIEW_3D', 'PREFERENCES'}:
                    area.tag_redraw()
    except Exception:                          # noqa: BLE001  界面状态不重要
        pass


def _poll_check_state():
    global _polling
    with _cache_lock:
        state = (_cache.get(REPO) or {}).get("status")
    if state == "checking":
        return 0.4
    _polling = False
    _redraw()
    return None


def _start_polling():
    global _polling
    if _polling:
        return
    try:
        _polling = True
        bpy.app.timers.register(_poll_check_state, first_interval=0.4)
    except Exception:                          # noqa: BLE001  没有计时器也能用
        _polling = False


def _online_access():
    """Blender 5 的「允许在线访问」总开关，关掉就一个请求都不发。"""
    return bool(getattr(bpy.app, "online_access", True))


def _start_check():
    if not _online_access():
        _store({"status": "no_access", "current_version": version_text()})
        _redraw()
        return False
    threading.Thread(target=_check_thread, name="gse-update-check", daemon=True).start()
    _start_polling()
    return True


def check_for_updates():
    """注册时调用：一个会话只查一次，后台线程。"""
    with _cache_lock:
        if _cache.get(REPO) is not None:
            return
    _start_check()


def force_check_for_updates():
    """面板上的刷新按钮：忽略缓存重查；返回是否真的发起了请求。"""
    return _start_check()


def stop():
    """插件卸载时收掉轮询计时器。"""
    global _polling
    _polling = False
    try:
        if bpy.app.timers.is_registered(_poll_check_state):
            bpy.app.timers.unregister(_poll_check_state)
    except Exception:                          # noqa: BLE001
        pass


def get_check_status():
    with _cache_lock:
        info = dict(_cache.get(REPO) or {})
    return {
        "status": info.get("status", "pending"),
        "current_version": info.get("current_version") or version_text(),
        "latest_version": info.get("latest_version", ""),
        "error": info.get("error", ""),
    }


def get_update_info():
    """有新版就返回信息，否则 None。"""
    with _cache_lock:
        info = dict(_cache.get(REPO) or {})
    if info.get("status") != "has_update":
        return None
    return {key: info.get(key, "")
            for key in ("current_version", "latest_version", "html_url", "zip_url")}


# ---------------------------------------------------------------- 覆盖安装
def _download(url, path, attempts=3, timeout=300):
    """下载到 path；成功返回 None，失败返回最后一次错误。"""
    last_error = "未知错误"
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with open(path, "wb") as handle:
                    shutil.copyfileobj(response, handle)
            return None
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except Exception as exc:               # noqa: BLE001  超时、断线都重试
            last_error = str(exc) or exc.__class__.__name__
        if attempt + 1 < attempts:
            time.sleep(1.5 * (attempt + 1))
    return last_error


def _find_extension_root(root):
    """找扩展本体那一层：含 blender_manifest.toml 的目录；找不到退而求其次找 __init__.py。

    两种发布包都能认：release 里的扩展包（根目录就是本体）和 GitHub 源码归档
    （repo-tag/gs_editor/...）。
    """
    fallback = None
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs
                         if name not in _SKIP_DIRS and not name.startswith("."))
        rel = os.path.relpath(folder, root)
        if rel != "." and rel.count(os.sep) >= 2:      # 只往下看两层，别扫到 C 盘去
            dirs[:] = []
            continue
        if "blender_manifest.toml" in files:
            return folder
        if fallback is None and "__init__.py" in files:
            fallback = folder
    return fallback


def _find_nested_zip(root):
    """解压出来的目录里如果只放了打包好的 zip，找出里面真含扩展的那个。"""
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs
                         if name not in _SKIP_DIRS and not name.startswith("."))
        for name in sorted(files):
            if not name.lower().endswith(".zip"):
                continue
            path = os.path.join(folder, name)
            try:
                with zipfile.ZipFile(path) as archive:
                    if any(entry.filename.endswith("blender_manifest.toml")
                           for entry in archive.infolist()):
                        return path
            except (zipfile.BadZipFile, OSError):
                continue
    return None


def _copy_extension(source, target):
    """把扩展目录里的文件覆盖进安装目录；返回覆盖的文件数。"""
    source = os.path.abspath(source)
    target = os.path.abspath(target)
    if source == target:
        raise ValueError("下载目录和安装目录是同一个")
    os.makedirs(target, exist_ok=True)
    copied = 0
    for folder, dirs, files in os.walk(source):
        dirs[:] = sorted(name for name in dirs
                         if name not in _SKIP_DIRS and not name.startswith("."))
        rel = os.path.relpath(folder, source)
        out_dir = target if rel == "." else os.path.join(target, rel)
        os.makedirs(out_dir, exist_ok=True)
        for name in sorted(files):
            if name.endswith(_SKIP_SUFFIXES):
                continue
            shutil.copy2(os.path.join(folder, name), os.path.join(out_dir, name))
            copied += 1
    # 旧的字节码留着只会让人怀疑「更新没生效」
    stale = os.path.join(target, "__pycache__")
    if os.path.isdir(stale):
        shutil.rmtree(stale, ignore_errors=True)
    return copied


def install_update(plugin_dir=None):
    """下载最新版并覆盖安装；返回 (成功, 提示文字)。plugin_dir 不给就装到本插件目录。"""
    with _cache_lock:
        info = dict(_cache.get(REPO) or {})
    if info.get("status") != "has_update":
        return False, "没有可安装的更新，先点「检查更新」"
    zip_url = info.get("zip_url", "")
    if not zip_url:
        return False, "发布里没找到下载地址"
    target = os.path.abspath(plugin_dir or ADDON_DIR)
    workdir = tempfile.mkdtemp(prefix="gs_editor_update_")
    try:
        archive_path = os.path.join(workdir, "update.zip")
        error = _download(zip_url, archive_path)
        if error:
            return False, f"下载失败：{error}"
        unpacked = os.path.join(workdir, "unpacked")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(unpacked)
        source = _find_extension_root(unpacked)
        for index in range(2):                 # 包里再套一个 zip 也能救回来
            if source is not None:
                break
            nested = _find_nested_zip(unpacked)
            if nested is None:
                break
            unpacked = os.path.join(workdir, f"nested{index}")
            with zipfile.ZipFile(nested) as archive:
                archive.extractall(unpacked)
            source = _find_extension_root(unpacked)
        if source is None:
            return False, "下载包里没找到插件文件，请手动更新"
        _copy_extension(source, target)
        with _cache_lock:
            if REPO in _cache:
                _cache[REPO]["status"] = "installed"
        return True, f"已更新到 v{info.get('latest_version', '')}，重启 Blender 生效"
    except Exception as exc:                   # noqa: BLE001  磁盘权限、坏包都归到这里
        return False, f"更新失败：{exc}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------- 操作符
class GSE_OT_check_update(bpy.types.Operator):
    """立即检查 GitHub 上的新版本"""

    bl_idname = "gse.check_update"
    bl_label = "检查更新"
    bl_description = "到 GitHub Releases 查一次新版本（后台进行，不会卡界面）"

    def execute(self, context):
        if not force_check_for_updates():
            self.report({'ERROR'}, "Blender 的在线访问已关闭：偏好设置 → 系统 → 允许在线访问")
            return {'CANCELLED'}
        self.report({'INFO'}, "正在后台检查更新…")
        return {'FINISHED'}


class GSE_OT_install_update(bpy.types.Operator):
    """下载 GitHub 上的最新版并覆盖安装，重启 Blender 后生效"""

    bl_idname = "gse.install_update"
    bl_label = "一键更新"
    bl_description = "下载最新版本，覆盖当前插件文件；重启 Blender 后生效"

    def execute(self, context):
        window = context.window
        if window is not None:
            window.cursor_set('WAIT')
        try:
            ok, message = install_update(plugin_dir=os.path.dirname(os.path.abspath(__file__)))
        finally:
            if window is not None:
                window.cursor_set('DEFAULT')
        self.report({'INFO'} if ok else {'ERROR'}, message)
        return {'FINISHED'} if ok else {'CANCELLED'}


classes = (GSE_OT_check_update, GSE_OT_install_update)
