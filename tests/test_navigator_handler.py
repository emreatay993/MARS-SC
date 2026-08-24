from types import SimpleNamespace

from ui.handlers import navigator_handler as navigator_module
from ui.handlers.navigator_handler import NavigatorHandler


def test_navigator_context_menu_exposes_native_file_actions(monkeypatch):
    index = SimpleNamespace(isValid=lambda: True)
    file_path = r"C:\work\result file.rst"
    model = SimpleNamespace(
        isDir=lambda _index: False,
        filePath=lambda _index: file_path,
    )
    tree = SimpleNamespace(
        indexAt=lambda _position: index,
        viewport=lambda: SimpleNamespace(mapToGlobal=lambda position: position),
    )
    actions = {}

    class Menu:
        def __init__(self, _parent):
            pass

        def addAction(self, text):
            action = SimpleNamespace(
                triggered=SimpleNamespace(connect=lambda callback: actions.setdefault(text, callback))
            )
            return action

        def exec_(self, _position):
            pass

    opened = []
    processes = []
    monkeypatch.setattr(navigator_module, "QMenu", Menu)
    monkeypatch.setattr(
        navigator_module.QDesktopServices,
        "openUrl",
        lambda url: opened.append(url.toLocalFile()) or True,
    )
    monkeypatch.setattr(
        navigator_module.subprocess,
        "Popen",
        lambda command: processes.append(command),
    )
    handler = NavigatorHandler(model, tree, SimpleNamespace())

    handler.show_navigator_context_menu(SimpleNamespace())
    for action in ("Show in File Explorer", "Open", "Open With..."):
        actions[action]()

    assert opened == [file_path.replace("\\", "/")]
    assert processes == [
        ["explorer.exe", f"/select,{file_path}"],
        ["rundll32.exe", "shell32.dll,OpenAs_RunDLL", file_path],
    ]
