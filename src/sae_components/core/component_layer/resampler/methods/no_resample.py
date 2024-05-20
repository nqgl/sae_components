from sae_components.core.component_layer.resampler.resampler import ResamplingMethod


class NoResampling(ResamplingMethod):
    def is_check_step(self):
        return False

    def is_resample_step(self):
        return False

    def resample_callback(self, cache, x=None, y_pred=None, y=None):
        raise Exception("NoResampling should not be called to resample")

    def get_directions(self, cache, x, y_pred, y):
        return super().get_directions(cache, x, y_pred, y)


# class LogOnlyResampling(ResamplingMethod):
