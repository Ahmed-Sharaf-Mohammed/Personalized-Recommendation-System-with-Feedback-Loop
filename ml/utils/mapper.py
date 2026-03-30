class DataMapper:
    def __init__(self):
        # interactions
        self.interaction_mapping = {
            "user_id": "user_id",
            "item_id": "parent_asin",
            "rating": "rating",
            "timestamp": "timestamp",
            "review_text": "text",
            "review_title": "title",
            "verified": "verified_purchase",
            "helpful_votes": "helpful_votes"
        }

        # items
        self.item_mapping = {
            "item_id": "parent_asin",
            "title": "title",
            "category": "main_category",
            "categories": "categories",
            "description": "description",
            "features": "features",
            "price": "price",
            "avg_rating": "average_rating",
            "rating_count": "rating_number",
            "store": "store",
            "details": "details"
        }

        # browsing logs
        self.browsing_mapping = {
            "user_id": "user_id",
            "item_id": "item_id",
            "event_type": "event_type",   # view, click, add_to_cart
            "timestamp": "timestamp",
            "session_id": "session_id",
            "device": "device",
            "source": "source"  # search, homepage, ad
        }

    def map_row(self, row, mapping):
        return {target: row.get(source) for target, source in mapping.items()}

    def map_interaction(self, row):
        return self.map_row(row, self.interaction_mapping)

    def map_item(self, row):
        return self.map_row(row, self.item_mapping)

    def map_browsing(self, row):
        return self.map_row(row, self.browsing_mapping)