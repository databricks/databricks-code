from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def _run_bootstrap():
    try:
        from coding_tool_gateway.bootstrap import main as bootstrap_main

        bootstrap_main()
    except Exception:
        # Runtime bootstrap is the reliable path. Install hooks are best effort only.
        pass


class InstallCommand(install):
    def run(self):
        super().run()
        _run_bootstrap()


class DevelopCommand(develop):
    def run(self):
        super().run()
        _run_bootstrap()


setup(cmdclass={"install": InstallCommand, "develop": DevelopCommand})
