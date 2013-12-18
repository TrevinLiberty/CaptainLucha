/****************************************************************************/
/* Copyright (c) 2013, Trevin Liberty
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
/****************************************************************************/
/*
 *	@author Trevin Liberty
 *	@file	
 *	@brief	
 *
/****************************************************************************/

#include "CLFileImporter.h"

#include <fstream>

namespace
{
	static const char* PASSWORD = "apple"; 
}

namespace CaptainLucha
{
	zipFile* CLFileImporter::m_zipArchive = NULL;
	LoadingType CLFileImporter::m_loadingType = CL_FOLDER_ZIP;

	CLFileImporter::CLFileImporter()
	{
		LoadStaticZipFromDisk();
	}

	CLFileImporter::~CLFileImporter()
	{

	}

	void CLFileImporter::DeleteStaticZipFromMemory()
	{
		unzClose(*m_zipArchive);
		delete m_zipArchive;
		m_zipArchive = NULL;
	}

	CLFile* CLFileImporter::LoadFile(const std::string& filePath)
	{
		ThreadLockGuard<ThreadMutex> guard(m_loadMutex);

		CLFile* newFile = NULL;
		if(m_loadingType == CL_ZIP || m_loadingType == CL_ZIP_FOLDER)
		{
			newFile = LoadFromZip(filePath);
		}
		if (m_loadingType == CL_ZIP_FOLDER && newFile == NULL)
		{
			newFile = LoadFromFile(filePath);
		}

		if(m_loadingType == CL_FOLDER || m_loadingType == CL_FOLDER_ZIP)
		{
			newFile = LoadFromFile(filePath);
		}
		if(m_loadingType == CL_FOLDER_ZIP && newFile == NULL)
		{
			newFile = LoadFromZip(filePath);
		}

		return newFile;
	}

	std::string CLFileImporter::GetLoadingTypeString() const
	{
		const char* result;
		switch(m_loadingType)
		{
		case CL_ZIP:
			result = "Zip Only";
		case CL_ZIP_FOLDER:
			result = "Zip -> Folder";
		case CL_FOLDER:
			result = "Folder Only";
		default:
			result = "Folder -> Zip";
		}

		return result;
	}

	void CLFileImporter::LoadStaticZipFromDisk()
	{
		if(!m_zipArchive)
		{
			m_zipArchive = new unzFile();
			*m_zipArchive = unzOpen("Data.zip");
		}
	}

	void CLFileImporter::SaveArchiveToDisk()
	{

	}

	CLFile* CLFileImporter::LoadFromFile(const std::string& filePath)
	{
		CLFile* newFile = NULL;
		std::ifstream inFile(filePath, std::ios::in | std::ios::binary);
		inFile >> std::noskipws;

		if(inFile.is_open())
		{
			newFile = new CLFile();
			std::streampos fileSize;
			fileSize = inFile.tellg();
			inFile.seekg(0, std::ios::end);
			fileSize = inFile.tellg() - fileSize;
			inFile.seekg(0, std::ios::beg);

			newFile->m_bufLength = static_cast<int>(fileSize);
			newFile->m_buffer = new char[newFile->m_bufLength];
			inFile.read(newFile->m_buffer, newFile->m_bufLength);
		}
		else
		{
			//LOAD ERROR
		}

		return newFile;
	}

	CLFile* CLFileImporter::LoadFromZip(const std::string& filePath)
	{
		CLFile* newFile = NULL;
		if(m_zipArchive)
		{
			int error = unzLocateFile(*m_zipArchive, filePath.c_str(), 0);
			if(!HandleAnyZIPErrors(error, "Locate File \"" + filePath + "\""))
			{
				error = unzOpenCurrentFilePassword(*m_zipArchive, PASSWORD);
				if(!HandleAnyZIPErrors(error, "Open File \"" + filePath + "\""))
				{
					unz_file_info info;
					unzGetCurrentFileInfo(*m_zipArchive, &info, NULL, 0, NULL, 0, NULL, 0);

					newFile = new CLFile();
					newFile->m_buffer = new char[info.uncompressed_size];

					error = unzReadCurrentFile(*m_zipArchive, (voidp)newFile->m_buffer, info.uncompressed_size);

					if(error > 0)
					{
						newFile->m_bufLength = error;
					}
					else if(error == 0)
					{
						newFile->m_bufLength = info.uncompressed_size;
						DebugBreak();
					}
					else
					{
						HandleAnyZIPErrors(error, "Load File");
						delete newFile;
						newFile = NULL;
					}

					unzCloseCurrentFile(*m_zipArchive);
				}

			}
		}
		return newFile;
	}

	bool CLFileImporter::HandleAnyZIPErrors(int error, const std::string& comment)
	{
		std::stringstream ss;
		ss << "ZIP ERROR: " << comment << ": ";

		if(error == UNZ_END_OF_LIST_OF_FILE)
		{
			ss << "\"END OF LIST OF FILE\"";
			traceWithFile(ss.str())
			return true;
		}
		else if(error == UNZ_ERRNO)
		{
			ss << "\"ERRNO\"";
			traceWithFile(ss.str())
			return true;
		}
		else if(error == UNZ_PARAMERROR)
		{
			ss << "\"PARAM ERROR\"";
			traceWithFile(ss.str())
			return true;
		}
		else if(error == UNZ_BADZIPFILE)
		{
			ss << "\"BAD ZIP FILE\"";
			traceWithFile(ss.str())
			return true;
		}
		else if(error == UNZ_INTERNALERROR)
		{
			ss << "\"INTERNAL ERROR\"";
			traceWithFile(ss.str())
			return true;
		}

		return false;
	}

	//////////////////////////////////////////////////////////////////////////
	//	Private
	//////////////////////////////////////////////////////////////////////////
}