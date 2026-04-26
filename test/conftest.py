import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "unittest: lightweight tests that run without GPU or pretrained weights")
