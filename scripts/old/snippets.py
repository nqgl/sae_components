def f(self):

    len([p for p in self.model.parameters()])
    self.model
    len(self.optim.param_groups)
    len(self.optim.param_groups[0]["params"])
    p = [p for p in self.model.parameters()]
    pg = [p.grad for p in self.model.parameters()]
    p[0].shape
    p[1]
    pg[1]
    print(pg)
    pg
    self.model.model.normalized.model.module.decoder.bias.shape
    pile.shape
    pile[i : i + batch_size].shape
    next(buffer).shape
