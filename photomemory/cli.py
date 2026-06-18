"""Command-line interface for the Photo Memory Generator."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from . import db
from .config import get_config

app = typer.Typer(add_completion=False, help="Generate iPhone-style memory videos from your NAS photos.")
console = Console()


@app.command()
def index(
    root: str = typer.Option(None, help="Only scan this root (e.g. 'triees' or 'amazon')."),
    subpath: str = typer.Option(None, help="Restrict to a subdirectory, e.g. a single year folder."),
):
    """Scan the NAS and (incrementally) populate the catalog."""
    from .scanner import scan

    stats = scan(root_filter=root, subpath=subpath)
    console.print(
        f"[green]Done.[/green] {stats['processed']:,} processed, "
        f"{stats['geocoded']:,} geocoded, {stats['candidates']:,} total files seen."
    )


@app.command()
def stats():
    """Show catalog summary."""
    db.init_db()
    with db.connect() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM media").fetchone()["c"]
        images = conn.execute("SELECT COUNT(*) c FROM media WHERE media_type='image'").fetchone()["c"]
        videos = conn.execute("SELECT COUNT(*) c FROM media WHERE media_type='video'").fetchone()["c"]
        with_gps = conn.execute("SELECT COUNT(*) c FROM media WHERE gps_lat IS NOT NULL").fetchone()["c"]
        with_place = conn.execute("SELECT COUNT(*) c FROM media WHERE place IS NOT NULL").fetchone()["c"]
        folders = conn.execute("SELECT COUNT(*) c FROM folders").fetchone()["c"]
        scored = conn.execute("SELECT COUNT(*) c FROM scores").fetchone()["c"]
        rng = conn.execute(
            "SELECT MIN(capture_dt) a, MAX(capture_dt) b FROM media WHERE capture_dt IS NOT NULL"
        ).fetchone()
        by_source = conn.execute(
            "SELECT date_source, COUNT(*) c FROM media GROUP BY date_source ORDER BY c DESC"
        ).fetchall()
        by_year = conn.execute(
            "SELECT substr(capture_dt,1,4) y, COUNT(*) c FROM media "
            "WHERE capture_dt IS NOT NULL GROUP BY y ORDER BY y DESC LIMIT 15"
        ).fetchall()

    t = Table(title="Catalog summary", show_header=False)
    t.add_row("Media files", f"{total:,}")
    t.add_row("  images / videos", f"{images:,} / {videos:,}")
    t.add_row("With GPS", f"{with_gps:,}")
    t.add_row("With place label", f"{with_place:,}")
    t.add_row("Folders", f"{folders:,}")
    t.add_row("Scored", f"{scored:,}")
    if rng and rng["a"]:
        t.add_row("Date range", f"{rng['a'][:10]} .. {rng['b'][:10]}")
    console.print(t)

    ds = Table(title="Date source")
    ds.add_column("source"); ds.add_column("count", justify="right")
    for r in by_source:
        ds.add_row(str(r["date_source"]), f"{r['c']:,}")
    console.print(ds)

    yt = Table(title="Recent years")
    yt.add_column("year"); yt.add_column("count", justify="right")
    for r in by_year:
        yt.add_row(str(r["y"]), f"{r['c']:,}")
    console.print(yt)


@app.command(name="list-months")
def list_months(year: int = typer.Argument(..., help="Year to summarize, e.g. 2024")):
    """List months of a year with their main events."""
    from .grouping import year_events

    months = year_events(year)
    if not months:
        console.print(f"[yellow]No photos found for {year}.[/yellow]")
        return
    for m, events in months.items():
        total = sum(e.count for e in events)
        top = sorted(events, key=lambda e: e.count, reverse=True)[:3]
        names = ", ".join(f"{e.label} ({e.count})" for e in top)
        console.print(f"[bold cyan]{year}-{m:02d}[/bold cyan]  {total:>4} photos  -  {names}")


@app.command(name="list-trips")
def list_trips(year: int = typer.Option(None, help="Restrict to a year.")):
    """List detected multi-day trips."""
    from .grouping import detect_trips

    trips = detect_trips(year=year)
    if not trips:
        console.print("[yellow]No trips detected.[/yellow]")
        return
    t = Table(title=f"Trips{f' in {year}' if year else ''}")
    t.add_column("#", justify="right"); t.add_column("dates"); t.add_column("days", justify="right")
    t.add_column("photos", justify="right"); t.add_column("place / title")
    for i, tr in enumerate(trips):
        t.add_row(str(i), f"{tr.date_start} -> {tr.date_end}", str(tr.span_days),
                  str(tr.count), tr.label)
    console.print(t)


@app.command(name="list-events")
def list_events(month: str = typer.Argument(..., help="Month as YYYY-MM, e.g. 2024-10")):
    """List the events (folders) within a month."""
    from .grouping import month_events

    year, mon = (int(x) for x in month.split("-"))
    events = month_events(year, mon)
    if not events:
        console.print(f"[yellow]No photos found for {month}.[/yellow]")
        return
    t = Table(title=f"Events in {month}")
    t.add_column("date"); t.add_column("photos", justify="right"); t.add_column("title"); t.add_column("place")
    for e in events:
        t.add_row(e.date_start, str(e.count), e.title or "-", e.place or "-")
    console.print(t)


build_app = typer.Typer(help="Build a memory video.")
app.add_typer(build_app, name="build")


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "memory"


def _render(memory, length, music, out_name):
    from .render import render_memory

    if len(memory) == 0:
        console.print("[yellow]No photos selected — nothing to render.[/yellow]")
        raise typer.Exit(1)
    out = get_config().output_dir / f"{out_name}.mp4"
    console.print(f"Rendering [cyan]{len(memory)}[/cyan] photos -> [green]{out}[/green]")
    render_memory(memory, out, music=music, length_seconds=length)
    console.print(f"[green]Done:[/green] {out}")


@build_app.command("month")
def build_month_cmd(
    month: str = typer.Argument(..., help="YYYY-MM, e.g. 2024-10"),
    length: float = typer.Option(None, help="Target video length in seconds."),
    no_music: bool = typer.Option(False, "--no-music", help="Render without audio."),
):
    """Build a monthly recap."""
    from .memory import build_month, target_count

    cfg = get_config().render
    length = length or cfg["default_length_seconds"]
    year, mon = (int(x) for x in month.split("-"))
    mem = build_month(year, mon, target_count(length, cfg["seconds_per_photo"]))
    _render(mem, length, not no_music, f"month_{month}")


@build_app.command("year")
def build_year_cmd(
    year: int = typer.Argument(..., help="Year, e.g. 2024"),
    length: float = typer.Option(None, help="Target video length in seconds."),
    no_music: bool = typer.Option(False, "--no-music", help="Render without audio."),
):
    """Build a yearly recap."""
    from .memory import build_year, target_count

    cfg = get_config().render
    length = length or max(cfg["default_length_seconds"], 120)
    mem = build_year(year, target_count(length, cfg["seconds_per_photo"]))
    _render(mem, length, not no_music, f"year_{year}")


@build_app.command("trip")
def build_trip_cmd(
    index: int = typer.Argument(None, help="Trip number from `list-trips`."),
    year: int = typer.Option(None, help="Year to search trips in."),
    around: str = typer.Option(None, help="Pick the trip covering this date (YYYY-MM-DD)."),
    length: float = typer.Option(None, help="Target video length in seconds."),
    no_music: bool = typer.Option(False, "--no-music", help="Render without audio."),
):
    """Build a trip recap (select by index, or by --around date)."""
    from .grouping import detect_trips
    from .memory import build_trip, target_count

    trips = detect_trips(year=year)
    if not trips:
        console.print("[yellow]No trips detected.[/yellow]")
        raise typer.Exit(1)
    trip = None
    if around:
        for t in trips:
            if t.date_start <= around <= t.date_end:
                trip = t
                break
    elif index is not None:
        if 0 <= index < len(trips):
            trip = trips[index]
    if trip is None:
        console.print("[red]Trip not found. Run `list-trips` to see options.[/red]")
        raise typer.Exit(1)

    cfg = get_config().render
    length = length or cfg["default_length_seconds"]
    mem = build_trip(trip, target_count(length, cfg["seconds_per_photo"]))
    _render(mem, length, not no_music, f"trip_{_slug(trip.label)}_{trip.date_start}")


faces_app = typer.Typer(help="Face recognition: enroll family members and boost their photos.")
app.add_typer(faces_app, name="faces")


@faces_app.command("extract")
def faces_extract(
    sample: int = typer.Option(0, help="Extract from a random sample of N images (0 = use scope)."),
    root: str = typer.Option(None, help="Only this root."),
    subpath: str = typer.Option(None, help="Restrict to a subdirectory (e.g. a year)."),
    force: bool = typer.Option(False, "--force", help="Re-extract even if already scanned."),
):
    """Detect+embed faces. Use --sample for quick enrollment, or scope by root/subpath."""
    from . import faceindex, db as _db

    if sample > 0:
        ids = faceindex.sample_media(sample)
    else:
        _db.init_db()
        clauses = ["media_type='image'"]
        params: list = []
        with _db.connect() as conn:
            if root:
                clauses.append("folder_id IN (SELECT id FROM folders WHERE source=?)")
                params.append(root)
            if subpath:
                clauses.append("path LIKE ?")
                params.append(f"%{subpath}%")
            rows = conn.execute(
                f"SELECT id FROM media WHERE {' AND '.join(clauses)}", params
            ).fetchall()
        ids = [r["id"] for r in rows]
    console.print(f"Extracting faces from [cyan]{len(ids):,}[/cyan] images...")
    n = faceindex.extract(ids, force=force)
    console.print(f"[green]Done.[/green] Processed {n:,} images.")


@faces_app.command("cluster")
def faces_cluster(
    eps: float = typer.Option(0.40, help="DBSCAN cosine distance threshold."),
    min_samples: int = typer.Option(5, help="Min faces to form a cluster."),
):
    """Cluster extracted faces and save review montages to output/people_review/."""
    from . import faceindex

    sizes = faceindex.cluster(eps=eps, min_samples=min_samples)
    if not sizes:
        console.print("[yellow]No faces to cluster. Run `faces extract` first.[/yellow]")
        return
    out = faceindex.save_montages()
    console.print(f"Found [cyan]{len(sizes)}[/cyan] clusters. Top sizes: "
                  + ", ".join(f"#{c}={n}" for c, n in list(sizes.items())[:10]))
    console.print(f"Review montages saved to [green]{out}[/green] — "
                  f"then label with: [bold]faces label <cluster#> \"Name\"[/bold]")


@faces_app.command("split")
def faces_split(
    cluster: int = typer.Argument(..., help="Cluster id to refine/split."),
    eps: float = typer.Option(0.24, help="Tighter DBSCAN distance for the split."),
    min_samples: int = typer.Option(4),
):
    """Re-cluster one over-merged cluster into separate people, then refresh montages."""
    from . import faceindex

    sizes = faceindex.split_cluster(cluster, eps=eps, min_samples=min_samples)
    if not sizes:
        console.print(f"[yellow]Cluster #{cluster} not found or empty.[/yellow]")
        return
    faceindex.save_montages()
    console.print(f"Split #{cluster} into {len(sizes)} sub-clusters: "
                  + ", ".join(f"#{c}={n}" for c, n in list(sizes.items())[:12]))
    console.print("Review the new cluster_*.jpg montages, then label them.")


@faces_app.command("label")
def faces_label(cluster: int = typer.Argument(...), name: str = typer.Argument(...)):
    """Assign a name to a cluster (look at output/people_review/cluster_XXX_*.jpg)."""
    from . import faceindex

    n = faceindex.label_cluster(cluster, name)
    console.print(f"[green]Labeled[/green] {n} faces in cluster #{cluster} as '{name}'.")


@faces_app.command("enroll")
def faces_enroll(
    name: str = typer.Argument(..., help="Person's name."),
    paths: list[str] = typer.Argument(..., help="Reference photo files and/or folders."),
    add: bool = typer.Option(False, "--add", help="Append to existing reference photos instead of replacing."),
):
    """Enroll a person from reference photos (reliable for people clustering can't separate)."""
    from . import faceindex

    n = faceindex.enroll_photos(name, paths, add=add)
    if n == 0:
        console.print(f"[yellow]No faces found in the given reference photos for '{name}'.[/yellow]")
    else:
        console.print(f"[green]Enrolled[/green] '{name}' from {n} reference face(s).")


@faces_app.command("birthday")
def faces_birthday(
    name: str = typer.Argument(..., help="Person's name (must match an enrolled person)."),
    date: str = typer.Argument(..., help="Birthdate as YYYY-MM-DD."),
):
    """Set a person's birthdate, enabling age-based disambiguation of look-alike siblings."""
    from . import faceindex

    faceindex.set_birthdate(name, date)
    console.print(f"[green]Set[/green] {name}'s birthdate to {date}.")


@faces_app.command("lifespan")
def faces_lifespan(
    name: str = typer.Argument(..., help="Enrolled person's name."),
    active_from: str = typer.Option(None, "--from", help="Earliest date they can appear (YYYY-MM-DD)."),
    until: str = typer.Option(None, "--until", help="Latest date they can appear (YYYY-MM-DD)."),
):
    """Bound the dates a person can appear in (e.g. --until for someone who has passed)."""
    from . import faceindex

    faceindex.set_active_range(name, active_from, until)
    console.print(f"[green]Set[/green] {name} active range from={active_from or '-'} until={until or '-'}.")


@faces_app.command("people")
def faces_people():
    """List enrolled people."""
    from . import faceindex

    people = faceindex.list_people()
    if not people:
        console.print("[yellow]No people enrolled yet.[/yellow]")
        return
    with db.connect() as conn:
        bdates = {r["name"]: r["birthdate"]
                  for r in conn.execute("SELECT name, birthdate FROM person_birth")}
    t = Table(title="Enrolled people")
    t.add_column("name"); t.add_column("faces", justify="right"); t.add_column("birthdate")
    for name, n in people:
        t.add_row(name, str(n), bdates.get(name, "-"))
    console.print(t)


@app.command()
def web(host: str = typer.Option("127.0.0.1", help="Bind address."),
        port: int = typer.Option(8000, help="Port.")):
    """Launch the local web UI (browse, preview/approve, render, watch)."""
    from .web import serve
    serve(host, port)


def main():
    app()


if __name__ == "__main__":
    main()
