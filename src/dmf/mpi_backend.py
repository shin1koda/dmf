def _worker_loop(world, images):
    """
    Worker loop for MPI ranks > 0.
    Each worker owns a full copy of `images`, each with calc attached.
    It receives:
        ("DO", {index: positions})
    and responds with:
        ("RESULT", {index: {"E": energy, "F": forces}})
    """

    rank = world.rank

    while True:
        cmd, payload = world.recv(source=0)

        if cmd == "STOP":
            #world.send(("STOPPED", None), dest=0)
            break

        elif cmd == "DO":
            results = {}

            # payload is a dict: {i: pos_i, j: pos_j, ...}
            for i, pos in payload.items():
                image = images[i]
                image.set_positions(pos)

                F = image.get_forces()
                E = image.get_potential_energy()

                results[i] = {"E": E, "F": F}

            world.send(("RESULT", results), dest=0)

        else:
            raise ValueError(
                f"Unknown MPI command {cmd!r} received by rank {rank}"
            )

