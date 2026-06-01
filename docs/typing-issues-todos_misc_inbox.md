    def loss(self, x, y, y_pred, cache: SAECache) -> torch.Tensor:
--> can get some types on x,y,y_pred?


can we do TruncatedLoss not as a mixin but as a wrapper that wraps a loss? I think that might be less nasty.


---


re: 
class Lambda(cl.Module):
    def __init__(self, func: Callable, module=None):


module seems like it could be Module|None, no? can you verify?

func seems like it should be Callable[[Tensor|DictBatch], Tensor|DictBatch]



---


    def __init__(
        self, target, scale: float = 1.0, increment=0.0003, multiplicative=True
    ):
yeah these args can also get types I think

---

    def penalty(self, x, *, cache) -> Tensor:

also here at least cache ought to have been a Cache or a SAECache?
I think there's been a systematic pattern of arguments not getting annotated in these first few passes? seems like it's been pretty consisten on return types. And of course some are still unclear, like x is still kind of unclear at the moment. I think Tensor|Sequence[Tensor]|DictBatch is an upper bound for the x in cl.Modules at the moment but they are in practice usually narrower.

