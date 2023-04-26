def make_ext(modname, pyxfilename):
    from distutils.extension import Extension

    ext = Extension(
        name=modname,
        sources=[pyxfilename],
    )

    return ext


def make_setup_args():
    return dict(script_args=["--verbose"])
