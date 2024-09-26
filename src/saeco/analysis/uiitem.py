from nicegui import ui
from saeco.analysis.ddmenuprop import ui_item


# class UIItemTest:
#     def __init__(self):
#         with ui.card():
#             self.item1

#     @ui_item(
#         lambda self, s: ui.input(
#             label="item1",
#             on_change=s,
#         )
#     )
#     def item1(self, e):
#         print(self.item1)
#         # if e is None:
#         #     return
#         # e.text = "item1"
#         return e.value


# def keyselect(self, s):
#     print("make keys")
#     sel = ui.select(
#         label="Keys", options=["key1", "key2", "key3"], on_change=s, multiple=True
#     )
#     sel.update()
#     return sel


# class KeysChoice:
#     def __init__(self):
#         with ui.row():
#             with ui.card():
#                 ui.label("x-keys")
#                 self.xkeys
#             with ui.card():
#                 ui.label("y-keys")
#                 self.ykeys

#     def update(self):
#         print(self.xkeys)

#     @ui_item(
#         # lambda s: ui.select(
#         #     label="Keys", options=["key1", "key2", "key3"], on_change=s, multiple=True
#         # )
#         keyselect
#     )
#     def xkeys(self, e):
#         return e.value

#     @ui_item(
#         lambda self, s: ui.select(
#             label="Keys", options=["key1", "key2", "key3"], on_change=s, multiple=True
#         )
#     )
#     def ykeys(self, e):
#         return e.value


# uii = UIItemTest()
# kc = KeysChoice()

# ui.run()


# %%
class Updating:
    def __init__(self, uii: "UIE", inst):
        self.uii = uii
        self.inst = inst

    def __enter__(self):
        self.uii._updating.add(id(self.inst))

    def __exit__(self, *args):
        self.uii._updating.remove(id(self.inst))

    def __contains__(self, inst):
        return inst in self.uii._updating


class UIE:
    def __init__(self, init_fn):

        self.init_fn = init_fn
        self.cls = None
        self.val = 0
        self._name = init_fn.__name__
        # self.fname = f"_{self._name}"
        self.ddname = f"_dd_{self._name}"
        self._on_set = None

        def default_getval(inst, el):
            try:
                return el.value
            except AttributeError:
                return ...

        self._updater_fn = None
        self._value_fn = default_getval
        self._updating = set()
        # print(dir(self.fn))

    def __set_name__(self, owner, name):
        self.cls = owner
        if not hasattr(self.cls, "__slots__"):
            self.cls.__slots__ = []
        if name not in self.cls.__slots__:
            self.cls.__slots__.append(name)
        print(owner, name)

    def updating(self, inst):
        assert id(inst) not in self._updating
        return Updating(self, inst)

    def updater(self, fn):
        self._updater_fn = fn
        return self

    def update_el(self, inst):
        if id(inst) in self._updating:
            return
        el = self.el(inst)
        with self.updating(inst):
            if self._updater_fn:
                self._updater_fn(inst, el)
            el.update()
            print("updated", type(el))

    def value(self, fn):
        self._value_fn = fn
        return self

    def get_value(self, inst):
        return self._value_fn(inst, self.el(inst))

    def __get__(self, instance, owner):
        if not hasattr(instance, self.ddname):
            # assert not hasattr(instance, self.ddname)
            # setattr(instance, self.fname, ...)
            self.init_el(instance)
        return self.get_value(instance)
        # return getattr(instance, self.fname)

        print("get")
        if self._updater_fn:
            print("update:", self._updater_fn(instance))

        return self.val + 6

    def el(self, inst):
        return getattr(inst, self.ddname)

    def on_set(self, fn):
        self._on_set = fn
        return self

    def __set__(self, instance, value):
        if self._on_set is None:
            raise NotImplementedError("no set")
        set_out = self._on_set(instance, self.el(instance), value)
        assert set_out is None

    def _update(self, inst):
        if hasattr(inst, "update"):
            inst.update()
        else:
            self.update_el(inst)

    def init_el(self, inst):
        def settr():
            self._update(inst)

        setattr(inst, self.ddname, self.init_fn(inst, settr))
        if hasattr(inst, "update"):
            upfn = getattr(inst, "update")
        else:
            upfn = lambda *a, **k: None

        def reup(*a, **k):
            self.update_el(inst)
            upfn(*a, **k)
            self.update_el(inst)

        setattr(inst, "update", reup)


if __name__ == "__main__":

    class Test:
        def __init__(self):
            self.v = 0
            with ui.card():
                self.label
                self.text

        def update(self): ...

        @UIE
        def f(self, cb):
            return ui.select(
                label="Keys",
                options=["key1", "key2", "key3"],
                multiple=True,
                on_change=cb,
            )

        @f.updater
        def f(self, e):
            print("up", e.value)

        @UIE
        def text(self, cb):
            return ui.input(label="item1", on_change=cb)

        @text.updater
        def text(self, e):
            self.v = e.value

        @UIE
        def label(self, cb):
            return ui.label("label")

        @label.updater
        def label(self, e):
            e.text = self.v

            # return "uppp"

    t = Test()
    # with ui.card():
    # t.f
    # t.f = 3
    t.f
    ui.run()
    # %%
