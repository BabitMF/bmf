import bmf.lib._hmp
from bmf.lib._bmf import sdk

ProcessDone = sdk.ProcessDone


class ModuleFunctor(object):

    def __init__(self, impl, itypes, otypes):
        self.impl = impl
        self.itypes = itypes
        self.otypes = otypes

    def _inputs(self, *args):
        if len(args) != len(self.itypes):
            raise ValueError("Expect {} args, got {}".format(
                len(self.itypes), len(args)))

        # type check
        ipkts = []
        for i, v in enumerate(args):
            if v is not None:
                if self.itypes[i] is not None and not isinstance(
                        v, self.itypes[i]):
                    raise ValueError(
                        "Expect {}th arg type is {}, got {}".format(
                            i, self.itype[i], type(v)))
            # else empty Packet will construct from None
            ipkts.append(sdk.Packet(v))

        return ipkts

    def __call__(self, *args):
        """
        interface for regular outputs
        """
        ipkts = self._inputs(*args)
        # process
        opkts = self.impl(ipkts)
        # cast to otypes
        outputs = []
        for i, v in enumerate(opkts):
            if self.otypes[i] is None:
                outputs.append(opkts[i])
            else:
                outputs.append(opkts[i].get(self.otypes[i]))
        return outputs

    def execute(self, *args, cleanup=True):
        """
        """
        ipkts = self._inputs(*args)
        self.impl.execute(ipkts, cleanup=cleanup)
        return self

    def fetch(self, idx):
        """
        """
        outputs = []
        for pkt in self.impl.fetch(idx):
            if self.otypes[idx] is None:
                outputs.append(pkt)
            else:
                outputs.append(pkt.get(self.otypes[idx]))
        return outputs


def make_sync_func(name: str,
                   itypes: list,
                   otypes: list,
                   type: str = "",
                   path: str = "",
                   entry: str = "",
                   option: dict = {},
                   node_id: int = 0):
    """
    name : module name
    itypes: expect input types, None means python type(ie. class, dict, list...)
    otypes: expect output types, None means python type
    type: module type(c++, go, python)
    path: module path
    option: Module options(dict)
    node_id: module node id
    """
    impl = sdk.ModuleFunctor(name=name,
                             type=type,
                             path=path,
                             entry=entry,
                             option=option,
                             node_id=node_id,
                             ninputs=len(itypes),
                             noutputs=len(otypes))
    return ModuleFunctor(impl, itypes, otypes)
