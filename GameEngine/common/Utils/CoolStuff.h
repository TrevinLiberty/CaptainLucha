#ifndef COOLSTUFF_H_CL
#define COOLSTUFF_H_CL

namespace CaptainLucha
{
	//returns the max unsigned int that numBits can represent
	inline int MaxUnsignedIntBits(int numBits)
	{
		return (1 << numBits) - 1;
	}
}

#endif