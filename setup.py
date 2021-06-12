from setuptools import Extension, setup

setup(
    packages=["predators_and_preys_env"],
    ext_modules=[
        Extension("game", ["predators_and_preys_env/engine/game.c"]),
        Extension("entity", ["predators_and_preys_env/engine/physics/entity.c"]),
    ],
)
