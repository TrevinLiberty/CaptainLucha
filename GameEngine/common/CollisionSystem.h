#ifndef COLLISION_SYSTEM_H_CL
#define COLLISION_SYSTEM_H_CL

#include "EventSystem/EventSystem.h"
#include "Math/Vector2D.h"

#include <vector>
#include <list>

namespace CaptainLucha
{
	class Actor;

	class CollisionSystem
	{
	public:
		static const int BUCKET_SIZE = 50;

		CollisionSystem(int mapWidth, int mapHeight);
		~CollisionSystem();

		void Update();

		void AddNewActor(NamedProperties& np);
		void RemoveUnit(NamedProperties& np);

	private:
		void InitBuckets();
		void ResolveCollisions();

		void ResolveCollision(Actor* actor1, Actor* actor2);
		inline int GetPositionBucketID(Vector2Df pos) const
		{
			pos.x_ /= BUCKET_SIZE;
			pos.y_ /= BUCKET_SIZE;

			if(pos.x_ < 0.0f && pos.x_ >= m_numBucketsWidth && pos.y_ < 0.0f && pos.y_ >= m_numBucketsHeight)
				return -1;

			return (int)floor(floor(pos.x_) * m_numBucketsHeight + floor(pos.y_));
		}

		inline int GetBucketIndex(int i, int j)
		{
			if(i < 0 && i >= m_numBucketsWidth && j < 0 && j >= m_numBucketsHeight)
				return -1;

			return i * m_numBucketsHeight + j;
		}

		void AddActorToBuckets(Actor* actor);
		void ComplexAddTobuckets(Actor* actor);

		int m_numBucketsWidth;
		int m_numBucketsHeight;

		std::vector<std::list<Actor*> > m_buckets;
		std::list<Actor*> m_registeredUnits;
	};
}

#endif