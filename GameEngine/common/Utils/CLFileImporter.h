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
 *	@file	CLFileImporter.h
 *	@brief	
 *
/****************************************************************************/

#ifndef FILE_ARCHIVE_H_CL
#define FILE_ARCHIVE_H_CL

#define ZLIB_WINAPI
#include "zip.h"
#include "unzip.h"

#include "UtilDebug.h"
#include "EventSystem/EventSystem.h"
#include "Threads/ThreadMutex.h"

#include <Utils/CommonIncludes.h>

namespace CaptainLucha
{
	enum LoadingType
	{
		CL_ZIP,
		CL_FOLDER,
		CL_ZIP_FOLDER,
		CL_FOLDER_ZIP
	};

	struct CLFile
	{
		char* m_buffer;
		int m_bufLength;

		CLFile() : m_buffer(NULL), m_bufLength(0) {}
		~CLFile(){delete [] m_buffer;}
	};

	class CLFileImporter
	{
	public:
		CLFileImporter();
		~CLFileImporter();

		//************************************
		// Method:    LoadStaticZipFromDisk
		// FullName:  CaptainLucha::CLFileImporter::LoadStaticZipFromDisk
		// Access:    public 
		// Returns:   void
		// Qualifier:
		//************************************
		void LoadStaticZipFromDisk();
		void DeleteStaticZipFromMemory();

		CLFile* LoadFile(const std::string& filePath);

		void SetLoadingType(LoadingType type) {m_loadingType = type;}

		std::string GetLoadingTypeString() const;

	protected:
		void SaveArchiveToDisk();

		CLFile* LoadFromFile(const std::string& filePath);
		CLFile* LoadFromZip(const std::string& filePath);

		//returns true if any errors occurred
		/**
		 * @brief     HandleAnyZIPErrors
		 * @param	  int error
		 * @param	  const std::string & comment
		 * @return    bool
		 */
		bool HandleAnyZIPErrors(int error, const std::string& comment);

	private:
		static unzFile* m_zipArchive;
		static LoadingType m_loadingType;

		ThreadMutex m_loadMutex;
	};
}

#endif