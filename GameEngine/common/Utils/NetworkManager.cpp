#include "NetworkManager.h"

#include "Utils/UtilDebug.h"

#pragma comment(lib, "ws2_32.lib")

namespace CaptainLucha
{
	NetworkSocket::NetworkSocket(CLCore& clCore)
		: m_clCore(clCore),
		  m_socket(INVALID_SOCKET)
	{
		if(!m_isInit)
		{
			WSADATA wsaData;
			int result = WSAStartup(MAKEWORD(2,2), &wsaData);
			PROMISES(result == 0 && "WSAStartUp FAILED")
		}
	}

	NetworkSocket::~NetworkSocket()
	{
		DeInitializeSocket();
	}

	void NetworkSocket::SendPacket(char* buffer, int size, const std::string& ip, int port)
	{
		REQUIRES(m_socket != INVALID_SOCKET)
		REQUIRES(buffer != NULL)
		REQUIRES(size > 0)

		sockaddr_in service;
		service.sin_family = AF_INET;
		service.sin_addr.s_addr = inet_addr(ip.c_str());
		service.sin_port = htons(port);

		int result = sendto(m_socket, buffer, size, 0, (sockaddr*)&service, sizeof(sockaddr_in));
	}

	int NetworkSocket::WaitForPacket(char* buffer, int size)
	{
		REQUIRES(buffer != NULL)
		REQUIRES(size > 0)

		sockaddr_in service;
		int length = sizeof(sockaddr_in);
		int result = recvfrom(m_socket, buffer, size, 0, (sockaddr*)&service, &length);

		std::stringstream ss;
		if(result > 0)
		{
			ss << "Bytes received: " << result;
			m_clCore.GetConsole().AddSuccessText(ss.str().c_str());
		}
		else if(result == 0)
		{
			ss << "Connection Closed!";
			m_clCore.GetConsole().AddErrorText(ss.str().c_str());
		}
		else
		{
			ss << "WaitForServerPacket Failed: " << WSAGetLastError();
			m_clCore.GetConsole().AddErrorText(ss.str().c_str());
		}

		return result;
	}

	void NetworkSocket::InitializeSocket(int port)
	{
		REQUIRES(m_socket == INVALID_SOCKET)

		m_socket = socket(AF_INET, SOCK_DGRAM, 0);
		PROMISES(m_socket != INVALID_SOCKET && "Unable to Create Socket")

		sockaddr_in service;
		const char* IPAdress = "127.0.0.1";
		service.sin_family = AF_INET;
		service.sin_addr.s_addr = inet_addr(IPAdress);
		service.sin_port = htons(port);

		int result = bind(m_socket, (SOCKADDR*)&service, sizeof(service));
		result = WSAGetLastError();
		PROMISES(result != SOCKET_ERROR && "Unable to Bind Socket")

		std::stringstream ss;
		m_clCore.GetConsole().AddSuccessText("Server successfully initialized!");
		ss << "    IP:   " << IPAdress;
		m_clCore.GetConsole().AddSuccessText(ss.str().c_str());
		ss.str(""); ss << "    Port: " << port; 
		m_clCore.GetConsole().AddSuccessText(ss.str().c_str());
	}

	void NetworkSocket::DeInitializeSocket()
	{
		if(m_socket != INVALID_SOCKET)
		{
			closesocket(m_socket);
			m_socket = INVALID_SOCKET;
		}

		PROMISES(m_socket == INVALID_SOCKET)
	}

	bool NetworkSocket::m_isInit  = false;
}