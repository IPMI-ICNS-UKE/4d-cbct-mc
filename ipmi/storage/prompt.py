from InquirerPy import inquirer


class text(inquirer.text):
    def __init__(
        self,
        *args,
        pre_fill: str = "",
        **kwargs,
    ):
        self._pre_fill = str(pre_fill) if pre_fill not in {"", None} else pre_fill
        super().__init__(*args, **kwargs)

    def pre_run(self):
        if self._pre_fill not in {"", None}:
            self._session.default_buffer.text = self._pre_fill
            self._session.default_buffer.cursor_position = len(self._pre_fill)

    def _run(self) -> str:
        return self._session.prompt(default=self._default, pre_run=self.pre_run)
