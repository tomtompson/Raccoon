

class HfReranker():
    def __init__ (
            self,
            model_id: str,
            config: dict | None = None
    ):
        self.config = config