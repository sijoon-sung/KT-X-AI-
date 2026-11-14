"""Check that each quantity in the GPT work order agrees with the route."""
from pathlib import Path
import sys
import unittest


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from product_live_worker import complete, route_pack


class WorkOrderAnswerTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            'moved_bikes': 6, 'distance_km': 4.33, 'work_minutes': 10.6,
            'observation': '2026-11-10T20:50:00+09:00',
            'stops': [
                dict(station_id='1', name='태평동 버드내초등학교', action='pickup', bikes=6),
                dict(station_id='2', name='문화동 센트럴파크(105동)', action='deliver', bikes=6),
            ],
        }
        self.ids = [fact['id'] for fact in route_pack(self.case)['facts']]

    def test_accepts_matching_order_and_amounts(self):
        text = '태평동 버드내초등학교 수거 6대\n문화동 센트럴파크(105동) 배치 6대'
        result = complete(self.case, {'answer': text, 'evidence_ids': self.ids})
        self.assertEqual(result['answer'], text)

    def test_rejects_second_quantity_that_differs_from_route(self):
        text = '태평동 버드내초등학교 수거 6대 또는 105대\n문화동 센트럴파크(105동) 배치 6대'
        with self.assertRaisesRegex(ValueError, '이동 수량'):
            complete(self.case, {'answer': text, 'evidence_ids': self.ids})
