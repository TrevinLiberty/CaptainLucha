///////////////////////////////////////////////////////////////////////
//
//	Author: Trevin Liberty
//	Data:   8/20/2013
//	Description:
//		
//
///////////////////////////////////////////////////////////////////////

#ifndef NETWORKMANAGER_H_CL
#define NETWORKMANAGER_H_CL

#define SERVER_PORT 80
#define CLIENT_PORT 81

#include <WinSock2.h>

#include "CLCore.h"
#include "Threads/Thread.h"
#include "Utils/CommonIncludes.h"

namespace CaptainLucha
{
	class NetworkSocket
	{
	public:
		NetworkSocket(CLCore& clCore);
		~NetworkSocket();

		void InitializeSocket(int port);
		void DeInitializeSocket();

		void SendPacket(char* buffer, int size, const std::string& ip, int port);
		int WaitForPacket(char* buffer, int size);

	protected:

	private:
		CLCore& m_clCore;

		SOCKET m_socket;

		std::vector<sockaddr_in> m_connectedClients;

		PREVENT_COPYING(NetworkSocket)

		static bool m_isInit;
	};
}

#endif