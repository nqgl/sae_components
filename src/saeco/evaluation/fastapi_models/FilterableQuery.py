from pydantic import BaseModel, Field


class FilterableQuery(BaseModel):
    filter_id: str | None = Field(
        default=None,
        description="ID of the filtered eval to use. None/null gets the unfiltered root eval.",
    )

    def filter(self, root):
        if self.filter_id is None:
            return root
        return root.open_filtered(self.filter_id)
