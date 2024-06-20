class Category(BaseModel):
    name: str


class CategSweep(BaseModel):
    category: Swept[Category]


cs = CategSweep(
    category=Swept[Category](
        values=[Category(name="VanillaSAE"), Category(name="HierarchicalSAE")]
    )
)
