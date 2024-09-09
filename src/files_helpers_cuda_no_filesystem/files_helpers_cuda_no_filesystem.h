//#include <filesystem>
#include <string>
//#include "opencv_utils.h"

class My_files
{
private:
	std::string images_directory_full_path;
	void create_directory_if_not_exists(std::string directory_full_path);

public:
	My_files();
	std::string get_directory_base_path();
	std::string get_image_full_path();
	void display_image();
	std::string getExecutablePath();
	bool directoryExists(const std::string& directory);
	std::string getParentPath(const std::string& filePath);

	std::string target_name;
		
};