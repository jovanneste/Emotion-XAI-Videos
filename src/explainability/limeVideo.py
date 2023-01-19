import lime
from lime import lime_base
from functools import partial
from functools import partial

class VideoExplanation(object):
    def __init__(self, video):
        self.video = video


class LimeVideoExplainer(object):
    def __init__(self):
        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))

        kernel_fn = partial(kernel, kernel_width=0.25)
        self.random_state = check_random_state(None)

        self.base = lime_base.LimeBase(kernel_fn, True, random_state=self.random_state)
