#ifndef WINDOWS_UTILS_H
#define WINDOWS_UTILS_H

#include <Windows.h>
#include <vector>
#include <string>

inline void GetSubDirs(
    std::vector<std::string>& output, 
    const std::string& path)
{
	WIN32_FIND_DATA findData;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	char fullpath[MAX_PATH];
	GetFullPathName(path.c_str(), MAX_PATH, fullpath, 0);
	std::string fp(fullpath);

	hFind = FindFirstFile((LPCSTR)(fp + "/*").c_str(), &findData);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do 
		{
			if ((findData.dwFileAttributes | FILE_ATTRIBUTE_DIRECTORY)
                    == FILE_ATTRIBUTE_DIRECTORY 
                && (findData.cFileName[0] != '.'))
			{
				output.push_back(findData.cFileName);
			}
		} 
		while (FindNextFile(hFind, &findData) != 0);
	}
}

inline void GetSubDirsRecursive(
    std::vector<std::string>& output, 
	const std::string& path,
	const std::string& prependStr)
{
	std::vector<std::string> firstLvl;
	GetSubDirs(firstLvl, path);
	for (std::vector<std::string>::iterator i = firstLvl.begin(); 
		i != firstLvl.end(); ++i)
	{
		output.push_back(prependStr + *i);
		GetSubDirsRecursive(output, 
			path + std::string("/") + *i + std::string("/"),
			prependStr + *i + std::string("/"));
	}
}

#endif