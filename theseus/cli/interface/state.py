"""Persistent state for user-defined interface views."""

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class View:
    """A saved collection of plots for one run."""

    id: int
    name: str


@dataclass(frozen=True)
class Plot:
    """A saved plot definition."""

    id: int
    name: str
    x: str | None
    y: str | None
    log_x: bool
    log_y: bool
    checkpoints: bool
    smoothing: float


class InterfaceState:
    """Store interface-only view definitions in an XDG state database."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            state_home = Path(
                os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
            )
            path = state_home / "theseus" / "ui.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS views (
                id INTEGER PRIMARY KEY,
                root TEXT NOT NULL,
                name_key TEXT NOT NULL,
                nonce TEXT NOT NULL,
                name TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS plots (
                id INTEGER PRIMARY KEY,
                view_id INTEGER NOT NULL REFERENCES views(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                x TEXT,
                y TEXT,
                log_x INTEGER NOT NULL DEFAULT 0,
                log_y INTEGER NOT NULL DEFAULT 0,
                checkpoints INTEGER NOT NULL DEFAULT 1,
                smoothing REAL NOT NULL DEFAULT 0
            );
            """
        )
        plot_columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(plots)")
        }
        if "smoothing" not in plot_columns:
            self.connection.execute(
                "ALTER TABLE plots ADD COLUMN smoothing REAL NOT NULL DEFAULT 0"
            )
        if "checkpoints" not in plot_columns:
            self.connection.execute(
                "ALTER TABLE plots ADD COLUMN checkpoints INTEGER NOT NULL DEFAULT 1"
            )

    def close(self) -> None:
        """Close the interface database."""
        self.connection.close()

    def views(
        self,
        root: Path,
        name: str | None = None,
        nonce: str | None = None,
    ) -> list[View]:
        """Return saved run or workspace views in creation order.

        Omitting both ``name`` and ``nonce`` addresses the one cross-cutting view
        collection owned by ``root``.
        """
        name, nonce = self._scope(name, nonce)
        rows = self.connection.execute(
            """
            SELECT id, name FROM views
            WHERE root = ? AND name_key = ? AND nonce = ?
            ORDER BY id
            """,
            (str(root.resolve()), name, nonce),
        )
        return [View(*row) for row in rows]

    def add_view(
        self,
        root: Path,
        name: str | None = None,
        nonce: str | None = None,
    ) -> View:
        """Create a view in a run or root workspace collection."""
        view_name = f"view {len(self.views(root, name, nonce)) + 1}"
        name, nonce = self._scope(name, nonce)
        cursor = self.connection.execute(
            "INSERT INTO views(root, name_key, nonce, name) VALUES (?, ?, ?, ?)",
            (str(root.resolve()), name, nonce, view_name),
        )
        self.connection.commit()
        return View(cursor.lastrowid or 0, view_name)

    def rename_view(self, view: View, name: str) -> View:
        """Persist and return a view's new name."""
        with self.connection:
            self.connection.execute(
                "UPDATE views SET name = ? WHERE id = ?", (name, view.id)
            )
        return View(view.id, name)

    def delete_view(self, view: View) -> None:
        """Delete a view and its plots in one transaction."""
        with self.connection:
            self.connection.execute("DELETE FROM views WHERE id = ?", (view.id,))

    def plots(self, view: View) -> list[Plot]:
        """Return a view's saved plots in creation order."""
        rows = self.connection.execute(
            """
            SELECT id, name, x, y, log_x, log_y, checkpoints, smoothing
            FROM plots WHERE view_id = ? ORDER BY id
            """,
            (view.id,),
        )
        return [
            Plot(
                row[0],
                row[1],
                row[2],
                row[3],
                bool(row[4]),
                bool(row[5]),
                bool(row[6]),
                row[7],
            )
            for row in rows
        ]

    def add_plot(self, view: View) -> Plot:
        """Create and return an empty plot in ``view``."""
        plot_name = f"plot {len(self.plots(view)) + 1}"
        cursor = self.connection.execute(
            "INSERT INTO plots(view_id, name) VALUES (?, ?)",
            (view.id, plot_name),
        )
        self.connection.commit()
        return Plot(
            cursor.lastrowid or 0, plot_name, None, None, False, False, True, 0.0
        )

    def rename_plot(self, plot: Plot, name: str) -> Plot:
        """Persist and return a plot's new name."""
        with self.connection:
            self.connection.execute(
                "UPDATE plots SET name = ? WHERE id = ?", (name, plot.id)
            )
        return Plot(
            plot.id,
            name,
            plot.x,
            plot.y,
            plot.log_x,
            plot.log_y,
            plot.checkpoints,
            plot.smoothing,
        )

    def delete_plot(self, plot: Plot) -> None:
        """Delete a plot in one transaction."""
        with self.connection:
            self.connection.execute("DELETE FROM plots WHERE id = ?", (plot.id,))

    def update_plot(
        self,
        plot: Plot,
        *,
        x: str | None,
        y: str | None,
        log_x: bool,
        log_y: bool,
        checkpoints: bool,
        smoothing: float,
    ) -> Plot:
        """Persist and return a plot's current axis configuration."""
        self.connection.execute(
            """
            UPDATE plots
            SET x = ?, y = ?, log_x = ?, log_y = ?, checkpoints = ?, smoothing = ?
            WHERE id = ?
            """,
            (x, y, log_x, log_y, checkpoints, smoothing, plot.id),
        )
        self.connection.commit()
        return Plot(plot.id, plot.name, x, y, log_x, log_y, checkpoints, smoothing)

    @staticmethod
    def _scope(name: str | None, nonce: str | None) -> tuple[str, str]:
        if (name is None) != (nonce is None):
            raise ValueError("view scope requires both name and nonce")
        return name or "", nonce or ""
