from abc import ABC

import numpy as np


class BaseDataProcessor(ABC):
    def __init__(self):
        self._chain = []

    def _chain_exe(self, image):
        for processor_step, kwargs in self._chain:
            if kwargs:
                image = processor_step(image, **kwargs)
            else:
                image = processor_step(image)
        return image.astype(np.float32)

    def add_processor_step(self, func_name, kwargs):
        func = getattr(self, func_name)
        self._chain.append((func, kwargs))

    @classmethod
    def to_chain(cls, chain_elems: dict):
        inst = cls()
        for chain_elem, elem_kwargs in chain_elems.items():
            inst.add_processor_step(chain_elem, elem_kwargs)
        return inst

    def __call__(self, img, **kwargs):
        return self._chain_exe(img)
