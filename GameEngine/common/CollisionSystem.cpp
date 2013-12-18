#include "CollisionSystem.h"
#include "Actors/Actor.h"
#include "Math/Vector2D.h"

namespace CaptainLucha
{
	//////////////////////////////////////////////////////////////////////////
	//	Public
	//////////////////////////////////////////////////////////////////////////
	CollisionSystem::CollisionSystem(int mapWidth, int mapHeight)
		: m_numBucketsHeight(mapHeight / BUCKET_SIZE),
		  m_numBucketsWidth(mapWidth / BUCKET_SIZE)
	{
		EventSystem::GetInstance()->RegisterObjectForEvent("AddCollision", *this, &CollisionSystem::AddNewActor);
		EventSystem::GetInstance()->RegisterObjectForEvent("ActorDead", *this, &CollisionSystem::RemoveUnit);

		m_buckets.resize(m_numBucketsWidth * m_numBucketsHeight);
	}

	CollisionSystem::~CollisionSystem()
	{

	}

	void CollisionSystem::Update()
	{
		InitBuckets();
		ResolveCollisions();
	}

	void CollisionSystem::AddNewActor(NamedProperties& np)
	{
		Actor* unit = NULL;

		np.Get("Actor", unit);
		m_registeredUnits.push_back(unit);
	}

	void CollisionSystem::RemoveUnit(NamedProperties& np)
	{
		Actor* deadActor;
		np.Get("Actor", deadActor);

		for (auto it = m_registeredUnits.begin(); it != m_registeredUnits.end();)
		{
			if((*it)->GetID() == deadActor->GetID())
			{
				it = m_registeredUnits.erase(it);
			}
			else
				++it;
		}
	}

	//////////////////////////////////////////////////////////////////////////
	//	Protected
	//////////////////////////////////////////////////////////////////////////
	void CollisionSystem::InitBuckets()
	{
		for (size_t i = 0; i < m_buckets.size(); ++i)
		{
			m_buckets[i].clear();
		}

		for(auto it = m_registeredUnits.begin(); it != m_registeredUnits.end(); ++it)
		{
			if((*it)->IsRelevant())
				AddActorToBuckets(*it);
		}
	}

	void CollisionSystem::ResolveCollisions()
	{
		for (size_t i = 0; i < m_buckets.size(); ++i)
		{
			std::list<Actor*>& units = m_buckets[i];
			
			for(auto outUnit = units.begin(); outUnit != units.end(); ++outUnit)
			{
				auto inUnit = outUnit;
				++inUnit;

				for (inUnit; inUnit != units.end(); ++inUnit)
				{
					ResolveCollision((*outUnit), (*inUnit));
				}
			}
		}
	}

	void CollisionSystem::ResolveCollision(Actor* actor1, Actor* actor2)
	{
		const float PUSH_AMOUNT = 0.65f;

		float radius1 = actor1->GetRadius();
		float radius2 = actor2->GetRadius();

		Vector2Df pos1(actor1->Get2DPos());
		Vector2Df pos2(actor2->Get2DPos());

		Vector2Df dir(pos1 - pos2);

		float sqrdDist = dir.SquaredLength();
		float radiusSqrd = radius1 + radius2;
		radiusSqrd *= radiusSqrd;

		if(sqrdDist < radiusSqrd)
		{
			float length = dir.Length();
			float mult = abs(1.0f - length / (radius1 + radius2));

			pos1 += dir * PUSH_AMOUNT * mult;
			if(actor1->CanBeMoved())
				actor1->SetPosition(Vector3Df(pos1.x_, 0.0f, pos1.y_));

			pos2 += dir * -PUSH_AMOUNT * mult;
			if(actor2->CanBeMoved())
				actor2->SetPosition(Vector3Df(pos2.x_, 0.0f, pos2.y_));

			actor1->Collided(actor2);
			actor2->Collided(actor1);
		}
	}

	void CollisionSystem::AddActorToBuckets(Actor* actor)
	{
		Vector2Df pos = actor->Get2DPos();
		float radius = actor->GetRadius();

		if(radius >= BUCKET_SIZE)
		{
			ComplexAddTobuckets(actor);
			return;
		}

		const int maxBucketSize = m_numBucketsHeight * m_numBucketsWidth; 

		int center = GetPositionBucketID(pos);
		int a = GetPositionBucketID(Vector2Df(pos.x_ - radius + 312.5f, pos.y_ - radius + 312.5f));
		int b = GetPositionBucketID(Vector2Df(pos.x_ + radius + 312.5f, pos.y_ - radius + 312.5f));
		int c = GetPositionBucketID(Vector2Df(pos.x_ + radius + 312.5f, pos.y_ + radius + 312.5f));
		int d = GetPositionBucketID(Vector2Df(pos.x_ - radius + 312.5f, pos.y_ + radius + 312.5f));

		if (center > -1 && center < maxBucketSize)
		{
			m_buckets[center].push_back(actor);
		}

		if(a > -1 && a < maxBucketSize && a != center)
		{
			m_buckets[a].push_back(actor);
		}

		if(b > -1 && b != a && b < maxBucketSize && b != center)
		{
			m_buckets[b].push_back(actor);
		}

		if(c > -1 && c != a && c != b && c < maxBucketSize && c != center)
		{
			m_buckets[c].push_back(actor);
		}

		if(d > -1 && d != b && d != c && d < maxBucketSize && d  != center)
		{
			m_buckets[d].push_back(actor);
		}
	}

	void CollisionSystem::ComplexAddTobuckets(Actor* actor)
	{
		Vector2Df pos = (actor->Get2DPos() + Vector2Df(312.5f, 312.5f)) / BUCKET_SIZE;
		float radius = actor->GetRadius() / BUCKET_SIZE;

		Vector2Df start(pos.x_ - radius, pos.y_ - radius);
		Vector2Df end(pos.x_ + radius, pos.y_ + radius);

		const int maxBucketSize = m_numBucketsHeight * m_numBucketsWidth; 

		for(int i = start.x_; i < end.x_; ++i)
		{
			for(int j = start.y_; j < end.y_; ++j)
			{
				int bucket = GetBucketIndex(i, j);
				m_buckets[bucket].push_back(actor);
			}
		}
	}
}