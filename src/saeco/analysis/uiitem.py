from nicegui import ui
from saeco.analysis.ddmenuprop import ui_item


class UIItemTest:
    def __init__(self):
        with ui.card():
            self.item1

    @ui_item(lambda s: ui.input(value="item1", on_change=s))
    def item1(self, e):
        print(self.item1)
        e.value = e.value + "!"
        e.update()
