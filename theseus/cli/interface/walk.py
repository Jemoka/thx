"""Browser-local Pygwalker exploration of the current data."""

from __future__ import annotations

from nicegui import run, ui

from theseus.cli.interface.data import DataFrameSource
from theseus.cli.interface.navigation import Button


class WalkButton(Button):
    def __init__(self, data: DataFrameSource) -> None:
        self.data = data
        super().__init__("walk", self.open)

    async def open(self) -> None:
        import pygwalker

        self.props("disabled")
        try:
            frame = await run.io_bound(self.data.to_dataframe)
            html = await run.io_bound(pygwalker.to_html, frame, appearance="light")
            with (
                ui.dialog().props("maximized") as dialog,
                ui.element("div").classes("walk-dialog"),
            ):
                Button("‹", dialog.close)
                iframe = ui.element("iframe").classes("grow w-full")
                iframe.props.update({"srcdoc": html, "title": "Pygwalker"})
            dialog.open()
        except Exception as error:
            ui.notify(f"Unable to launch Pygwalker: {error}", type="negative")
        finally:
            self.props(remove="disabled")
