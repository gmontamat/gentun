import json
from unittest.mock import MagicMock, patch

import pytest

from src.gentun.models.base import Handler
from src.gentun.services import RedisController, RedisWorker


class MockHandler(Handler):
    def __init__(self, param1: int, param2: str = "default"):
        self.param1 = param1
        self.param2 = param2

    def evaluate(self, x_train, y_train):
        return 0.9


@patch("src.gentun.services.redis.StrictRedis")
def test_redis_controller_init(mock_redis):
    controller = RedisController("test", timeout=10)
    assert controller.name == "test"
    assert controller.host == "localhost"
    assert controller.port == 6379
    assert controller.client == mock_redis.return_value
    assert controller.job_queue == "job_queue"
    assert controller.results_queue == "results_queue"
    assert controller.timeout == 10


@patch("src.gentun.services.redis.StrictRedis")
def test_redis_controller_get_worker_details(mock_redis):
    # localhost
    controller = RedisController("test")
    details = controller.get_worker_details(MockHandler)
    assert "from gentun.models.test_services import MockHandler" in details
    # random ip
    controller = RedisController("test", host="192.168.0.100")
    details = controller.get_worker_details(MockHandler)
    assert "from gentun.models.test_services import MockHandler" in details
    assert "192.168.0.100" in details


@patch("src.gentun.services.redis.StrictRedis")
def test_redis_controller_send_job(mock_redis):
    controller = RedisController("test")
    # Send first job
    job_id = controller.send_job(MockHandler, param1=1, param2="value")
    assert isinstance(job_id, str)
    job = json.loads(mock_redis.return_value.rpush.call_args[0][1])
    assert job["name"] == "test"
    assert job["handler"] == "MockHandler"
    assert job["kwargs"] == {"param1": 1, "param2": "value"}
    # Send a second job
    job_id = controller.send_job(MockHandler, param1=2, param2="value2")
    assert isinstance(job_id, str)
    job = json.loads(mock_redis.return_value.rpush.call_args[0][1])
    assert job["name"] == "test"
    assert job["handler"] == "MockHandler"
    assert job["kwargs"] == {"param1": 2, "param2": "value2"}


@patch("src.gentun.services.redis.StrictRedis")
def test_redis_controller_wait_for_result(mock_redis):
    controller = RedisController("test")
    job_id = "test_job_id"
    result = {"id": job_id, "name": "test", "fitness": 0.9}
    ignore_result = {"id": "not_test_job_id", "name": "test", "fitness": 0.9}
    mock_redis.return_value.lpop.side_effect = [None, json.dumps(ignore_result), json.dumps(result)]
    fitness = controller.wait_for_result(job_id)
    assert fitness == 0.9


@patch("src.gentun.services.redis.StrictRedis")
def test_redis_controller_wait_for_result_timeout(mock_redis):
    controller = RedisController("test", timeout=1)
    job_id = "test_job_id"
    mock_redis.return_value.lpop.return_value = None
    with pytest.raises(TimeoutError):
        controller.wait_for_result(job_id)


@patch("src.gentun.services.redis.StrictRedis")
def test_redis_worker_init(mock_redis):
    worker = RedisWorker("test", MockHandler)
    assert worker.name == "test"
    assert worker.handler == MockHandler
    assert worker.client == mock_redis.return_value
    assert worker.job_queue == "job_queue"
    assert worker.results_queue == "results_queue"
    assert worker.timeout == 259200


@patch("src.gentun.services.redis.StrictRedis")
def test_redis_worker_process_job(mock_redis):
    worker = RedisWorker("test", MockHandler)
    fitness = worker.process_job([1, 2, 3], [4, 5, 6], param1=1, param2="value")
    assert fitness == 0.9


@patch("src.gentun.services.redis.StrictRedis")
def test_redis_worker_run(mock_redis):
    worker = RedisWorker("test", MockHandler)
    job_data = {
        "id": "test_job_id",
        "name": "test",
        "handler": "MockHandler",
        "kwargs": {"param1": 1, "param2": "value"},
    }
    ignore_job_data = {
        "id": "not_test_job_id",
        "name": "test",
        "handler": "NotMockHandler",
        "kwargs": {"param1": 1, "param2": "value"},
    }
    mock_redis.return_value.lpop.side_effect = [json.dumps(ignore_job_data)] + [json.dumps(job_data)] + [None]
    with patch.object(worker, "process_job", return_value=0.9) as mock_process_job:
        with patch("time.sleep", side_effect=KeyboardInterrupt):
            worker.run([1, 2, 3], [4, 5, 6])
            mock_process_job.assert_called_once_with([1, 2, 3], [4, 5, 6], param1=1, param2="value")
            result = json.loads(mock_redis.return_value.rpush.call_args[0][1])
            assert result["id"] == "test_job_id"
            assert result["name"] == "test"
            assert result["fitness"] == 0.9
