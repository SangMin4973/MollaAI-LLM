from __future__ import annotations

import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

import server


class MemoryPointsEndpointTests(unittest.TestCase):
    def test_upserts_embedded_points_into_memory_store(self) -> None:
        mock_store = Mock()
        original_store = server.memory_store
        server.memory_store = mock_store
        client = TestClient(server.app)

        try:
            response = client.post(
                "/memory/points",
                json={
                    "points": [
                        {
                            "id": "uuid-1",
                            "vector": [0.1, 0.2, 0.3],
                            "payload": {
                                "userId": "user-123",
                                "phoneNumber": "01012345678",
                                "userText": "I received the wrong item.",
                                "assistantText": "I'm sorry to hear that. What item did you expect?",
                                "createdAt": "2026-05-20T07:08:33.742000Z",
                                "audioKey": "calls/a/turns/5.wav",
                            },
                        }
                    ]
                },
            )
        finally:
            server.memory_store = original_store

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "count": 1})
        mock_store.upsert_points.assert_called_once()
        stored_points = mock_store.upsert_points.call_args.args[0]
        self.assertEqual(stored_points[0]["payload"]["userText"], "I received the wrong item.")
        self.assertEqual(stored_points[0]["payload"]["audioKey"], "calls/a/turns/5.wav")

    def test_allows_nullable_user_id_and_assistant_text(self) -> None:
        mock_store = Mock()
        original_store = server.memory_store
        server.memory_store = mock_store
        client = TestClient(server.app)

        try:
            response = client.post(
                "/memory/points",
                json={
                    "points": [
                        {
                            "id": "uuid-1",
                            "vector": [0.1, 0.2, 0.3],
                            "payload": {
                                "userId": None,
                                "phoneNumber": "01012345678",
                                "userText": "I received the wrong item.",
                                "assistantText": None,
                                "createdAt": "2026-05-20T07:08:33.742000Z",
                                "audioKey": "calls/a/turns/5.wav",
                            },
                        }
                    ]
                },
            )
        finally:
            server.memory_store = original_store

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "count": 1})
        stored_points = mock_store.upsert_points.call_args.args[0]
        self.assertIsNone(stored_points[0]["payload"]["userId"])
        self.assertIsNone(stored_points[0]["payload"]["assistantText"])


if __name__ == "__main__":
    unittest.main()
