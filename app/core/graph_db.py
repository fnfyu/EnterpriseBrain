from neo4j import AsyncGraphDatabase
from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    return _driver


async def close_driver():
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


async def init_graph_schema():
    """Create indexes and constraints."""
    driver = get_driver()
    async with driver.session() as session:
        await session.run(
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE"
        )
        await session.run(
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)"
        )
        logger.info("Neo4j schema initialized")


class GraphStore:
    def __init__(self):
        self.driver = get_driver()

    async def upsert_entity(self, entity_id: str, name: str, entity_type: str, properties: dict = None):
        async with self.driver.session() as session:
            await session.run(
                """
                MERGE (e:Entity {id: $id})
                SET e.name = $name, e.type = $type, e.props = $props
                """,
                id=entity_id, name=name, type=entity_type, props=properties or {},
            )

    async def upsert_relation(self, from_id: str, to_id: str, rel_type: str, properties: dict = None):
        async with self.driver.session() as session:
            await session.run(
                """
                MATCH (a:Entity {id: $from_id}), (b:Entity {id: $to_id})
                MERGE (a)-[r:RELATES {type: $rel_type}]->(b)
                SET r.props = $props
                """,
                from_id=from_id, to_id=to_id, rel_type=rel_type, props=properties or {},
            )

    async def search_by_entities(self, entity_names: list[str], limit: int = 10) -> list[dict]:
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity)
                WHERE e.name IN $names
                OPTIONAL MATCH (e)-[r]->(related)
                RETURN e.id AS id, e.name AS name, e.type AS type,
                       collect({rel: r.type, target: related.name}) AS relations
                LIMIT $limit
                """,
                names=entity_names, limit=limit,
            )
            return [dict(record) async for record in result]
