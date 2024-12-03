from setuptools import setup, find_packages


with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()


setup(
    name="Telegram_News_SemAn",
    author="Butera Gaetano Antonio",
    description="The programs provides an analysis tool for telegram news chanel. It takes html files of said channels "
                "and returns a csv file with information about the messages.",
    version='0.1',
    package_dir={"": "src"},
    install_requires=requirements,
    dependency_links=[
        'https://download.pytorch.org/whl/cpu'
    ],
    entry_points={
        "console_scripts": [
            "SemAn = entry:main"
            ]
        },
)