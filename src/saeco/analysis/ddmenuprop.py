# from saeco.analysis.update_render import render_update_list, update_render


from nicegui import ui

dd_update_list = []


def ddupdate():
    for update in dd_update_list:
        update()


def ddmenuprop(optfn):

    # def wrap(optfn):
    # val = options[0]

    _name = optfn.__name__
    fname = f"_{_name}"
    ddname = f"_dd_{_name}"
    prev_optname = f"_opts_prev_{_name}"

    def dynfield(fstr):
        def rw(self, setto=None):
            if setto is None:
                return getattr(self, fstr)
            else:
                setattr(self, fstr, setto)

        return rw

    dd = dynfield(ddname)
    prev_opts = dynfield(prev_optname)
    val = dynfield(fname)
    # def prev_opts(self, setto=None):
    #     if setto is None:
    #         return getattr(self, prev_optname)
    #     else:
    #         setattr(self, prev_optname, setto)

    def getter(self):
        # nonlocal val
        # return val
        if not hasattr(self, fname):
            assert not hasattr(self, ddname)
            setattr(self, fname, ...)
            init_prop(self)
            setattr(self, fname, optfn(self)[0])
        return getattr(self, fname)

    def setter(self, value):
        setattr(self, fname, value)
        # nonlocal val
        # val = value
        # fn(self)

        dd(self).text = f"{_name}={value}"
        dd(self).update()
        # update_render()

    prop = property(getter, setter)

    def setvalue(self, value):
        # nonlocal val

        def setter():
            setattr(self, fname, value)
            dd(self).text = f"{_name}={value}"
            dd(self).update()
            print(optfn)
            optfn(self)
            ui.update()
            if hasattr(self, "update"):
                self.update()
            # update_render()

        return setter

    def update_dd(self):
        if not hasattr(self, prev_optname):
            setattr(self, prev_optname, None)
        dd(self).text = f"{_name}={val(self)}"

        prev_options = prev_opts(self)
        options = optfn(self)
        if prev_options != options:
            dd(self).clear()
            with dd(self):
                for opt in options:
                    ui.item(repr(opt), on_click=setvalue(self, opt))
        prev_options = options

    def init_prop(self):
        dd_obj = ui.dropdown_button(f"{_name}", auto_close=True)
        dd_obj._event_listeners
        dd(self, dd_obj)
        update_dd(self)
        dd_update_list.append(lambda: update_dd(self))
        # def setvalue(value):
        #     # nonlocal val

        #     def setter():
        #         setattr(self, fname, value)
        #         dd.text = f"{_name}={value}"
        #         dd.update()
        #         print(optfn)
        #         optfn(self)
        #         update_render()

        #     return setter

    return prop


def ui_item(lmda):
    def wrap(optfn):
        _name = optfn.__name__
        fname = f"_{_name}"
        ddname = f"_dd_{_name}"

        def dynfield(fstr):
            def rw(self, setto=None):
                if setto is None:
                    return getattr(self, fstr)
                else:
                    setattr(self, fstr, setto)

            return rw

        dd = dynfield(ddname)
        val = dynfield(fname)

        def getter(self):
            if not hasattr(self, fname):
                assert not hasattr(self, ddname)
                setattr(self, fname, ...)
                init_prop(self)
                # setattr(self, fname, optfn(self)[0])
            return getattr(self, fname)

        def setter(self, value):
            setattr(self, fname, value)

            dd(self).text = f"{_name}={value}"
            dd(self).update()

        prop = property(getter, setter)

        # def setvalue(self, value):

        #     def setter():
        #         setattr(self, fname, value)
        #         dd(self).text = f"{_name}={value}"
        #         dd(self).update()
        #         print(optfn)
        #         optfn(self)
        #         ui.update()
        #         if hasattr(self, "update"):
        #             self.update()

        #     return setter

        def update_dd(self):
            if not hasattr(self, prev_optname):
                setattr(self, prev_optname, None)

        def init_prop(self):
            def settr():
                setattr(self, fname, optfn(self, dd(self)))
                ui.update()
                if hasattr(self, "update"):
                    self.update()

            dd_obj = lmda(self, settr)
            dd_obj._event_listeners
            dd(self, dd_obj)
            # update_dd(self)
            dd_update_list.append(lambda: update_dd(self))
            # def setvalue(value):
            #     # nonlocal val

            #     return setter

        update = None

        def set_update(up):
            nonlocal update
            update = up

        object.__setattr__(prop, "update", set_update)
        # setattr(prop, "updater", set_update)
        return prop

    return wrap
