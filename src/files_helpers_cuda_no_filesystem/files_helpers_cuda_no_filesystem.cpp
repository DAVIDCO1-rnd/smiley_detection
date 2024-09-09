
#include <iostream>
#include <Windows.h>
#include "files_helpers_cuda_no_filesystem.h"
#include <sys/stat.h> // For mkdir
#ifdef _WIN32
#include <direct.h> // For _mkdir on Windows
#endif

std::string My_files::getExecutablePath() {
    char buffer[MAX_PATH];
    GetModuleFileName(NULL, buffer, MAX_PATH);
    return std::string(buffer);
}

std::string My_files::getParentPath(const std::string& filePath) {
    size_t found = filePath.find_last_of("/\\");
    if (found != std::string::npos) {
        return filePath.substr(0, found); // Extract the parent path
    }
    return ""; // No parent found
}

My_files::My_files()
{

}

std::string My_files::get_directory_base_path()
{
    char buffer[MAX_PATH];
    GetModuleFileName(NULL, buffer, MAX_PATH);
    std::string fullPath(buffer);
    size_t index_of_last_slash = fullPath.find_last_of("\\/");
    size_t full_path_length = fullPath.length();
    std::string target_name_with_exe_extension = fullPath.substr(index_of_last_slash + 1, full_path_length);
    this->target_name = target_name_with_exe_extension.substr(0, target_name_with_exe_extension.length() - 4);
    std::string output_directory_str = fullPath.substr(0, index_of_last_slash);
    std::string parent1_path = getParentPath(output_directory_str);
    std::string parent2_path = getParentPath(parent1_path);
    std::string parent3_path = getParentPath(parent2_path);
    std::string directory_base_path = getParentPath(parent3_path);
    return directory_base_path;    
}

bool My_files::directoryExists(const std::string& directory) {
#ifdef _WIN32
    DWORD attrib = GetFileAttributes(directory.c_str());
    return (attrib != INVALID_FILE_ATTRIBUTES && (attrib & FILE_ATTRIBUTE_DIRECTORY));
#else
    if (access(directory.c_str(), F_OK) != -1) {
        // Directory exists
        return true;
    }
    else {
        // Directory does not exist or cannot be accessed
        return false;
    }
#endif
}

//void My_files::create_directory_if_not_exists(std::string directory_full_path)
//{
//    if (!std::filesystem::exists(directory_full_path)) {
//        // Create the directory
//        if (std::filesystem::create_directory(directory_full_path)) {
//            std::cout << "Directory " << directory_full_path << " created successfully." << std::endl;
//        }
//        else {
//            std::cerr << "Failed to create directory." << directory_full_path << std::endl;
//        }
//    }
//}

void My_files::create_directory_if_not_exists(std::string directory_full_path) {
    bool is_directory_exist = directoryExists(directory_full_path);
    if (!is_directory_exist)
    {
#ifdef _WIN32
        int mkdir_status = _mkdir(directory_full_path.c_str());
#else
        int mkdir_status = mkdir(directory.c_str(), 0777);
#endif
        if (mkdir_status != 0)
        {
            std::cerr << "Failed to create directory: " << directory_full_path << std::endl;
        }       
    }
}


