#include <iostream>

#include "Convert.h"
#include "WindowsUtils.h"

static const char* g_IMPORT_PATH = "Data\\Import\\";
static const char* g_EXPORT_PATH = "Data\\Export\\";

void OutputErrorAndExit(const char* errorMsg);
bool HasValidExtension(std::string path);
void ConvertFilesInDirectory(const std::string& path);

using namespace CaptainLucha;

int main()
{
	std::vector<std::string> allDirectories;
	GetSubDirsRecursive(allDirectories, g_IMPORT_PATH, "");

	for(size_t  i = 0; i < allDirectories.size(); ++i)
	{
		ConvertFilesInDirectory(g_IMPORT_PATH + allDirectories[i] + "\\*");
	}
}	

void OutputErrorAndExit(const char* errorMsg)
{
	MessageBox(NULL, errorMsg, NULL, MB_OK);
	system("PAUSE");
	return;
}

//Currently tested and working extensions.
const int NUM_VALID_EXTENSIONS = 4;
std::string validExtensions[NUM_VALID_EXTENSIONS] =
{
	".obj",
	".blend",
	".3ds",
	".ase"
};

bool HasValidExtension(std::string path)
{
	std::string::size_type idx = path.rfind('.');

	if(idx != std::string::npos)
	{
		std::string extension = path.substr(idx);

		for(int i = 0; i < NUM_VALID_EXTENSIONS; ++i)
		{
			if(extension.size() > 0 && _strcmpi(extension.c_str(), validExtensions[i].c_str()) == 0)
			{
				return true;
			}
		}
	}

	return false;
}

//If the file at path is a valid model, try to convert it.
void ConvertFilesInDirectory(const std::string& path)
{
	WIN32_FIND_DATA ffd;
	HANDLE file = INVALID_HANDLE_VALUE;

	file = FindFirstFile(path.c_str(), &ffd);

	if(file == INVALID_HANDLE_VALUE)
		return;

	do 
	{
		if(HasValidExtension(ffd.cFileName))
		{
			std::cout << "Converting File: " << ffd.cFileName << std::cout;
			std::string fullPath = path;
			fullPath.resize(fullPath.size() - 1);
			fullPath = fullPath + ffd.cFileName;
			ConvertAndExportFile(fullPath.c_str(), g_EXPORT_PATH);
		}
		else if(HasValidExtension(ffd.cAlternateFileName))
		{
			std::cout << "Converting File: " << ffd.cAlternateFileName << std::cout;
			ConvertAndExportFile(ffd.cAlternateFileName, g_EXPORT_PATH);
		}

	} while (FindNextFile(file, &ffd) != 0);
}