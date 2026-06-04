from base import BaseService, Helper


class AppService(BaseService):
    def run(self) -> str:
        helper = Helper()
        return helper.ping() + super().run()
