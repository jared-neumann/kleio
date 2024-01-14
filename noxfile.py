import nox

@nox.session(python=["3.11"])
def tests(session):
    session.run("poetry", "install", external=True)
    session.run("pytest", "--cov")

@nox.session(python="3.11")
def black(session):
    session.run("black", "src")

@nox.session(python="3.11")
def lint(session):
    session.run("flake8", "src")