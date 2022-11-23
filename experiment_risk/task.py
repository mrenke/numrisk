import argparse
from session import RiskPileSession


def main():

    session = RiskPileSession('test', settings_file='settings/macbook.yml')
    session.create_trials()

    session.run()
    session.close()

if __name__ == '__main__':
    main()