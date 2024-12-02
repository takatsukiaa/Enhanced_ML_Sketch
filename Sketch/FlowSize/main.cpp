#include "CMSketch.h"
#include "CMSACSketch.h"
#include "CUSketch.h"
#include "CUSACSketch.h"
#include "PSketch.h"
#include "PSACSketch.h"
#include <fstream>

CMSketch *Sketch = new CMSketch(3, 2810);
std::unordered_map<std::string, uint> actual_size;

int main(){
	/*Insert your code here using the flowsize interface*/

	std::ifstream file("binary.dat",std::ios::binary);
	if(!file){
		std::cout<<"error, file not found!\n";
		return -1;
	}
	unsigned char* buffer = new unsigned char [13];
	memset(buffer,0,13);
	while(file.read(reinterpret_cast<char*>(buffer),13)|| file.gcount() > 0)
	{
		std::string data(reinterpret_cast<char*>(buffer), file.gcount());
		cuc* constData = buffer;
		Sketch->Enhanced_Insert(constData);
		actual_size[data]++;
	}
	file.close();
	file.open("binary.dat",std::ios::binary);
	memset(buffer,0,13);
	FILE* out = fopen("result.txt", "w");
	FILE* counters = fopen("counters.txt","w");
	FILE* all_flows = fopen("flows.txt","w");
 	for (auto it = actual_size.begin(); it != actual_size.end(); ++it) 
	{
        cuc* temp = reinterpret_cast<cuc*>(const_cast<char*>(it->first.c_str()));
		int second = it->second;
		Sketch->PrintCounterFile(temp, second, all_flows);
    }
	cuc* path = (unsigned char*)"parameter.txt";
	Sketch->LoadPara(path);
	float AAE = 0;
	float ARE = 0;
	while(file.read(reinterpret_cast<char*>(buffer),13)|| file.gcount() > 0)
	{
		std::string data(reinterpret_cast<char*>(buffer), 13);
		cuc* constData = buffer;
		uint a = actual_size[data];
		uint query_val;
		query_val = Sketch->Query(constData);
		AAE+=Sketch->CalculateAAE(constData,actual_size[data]);
		// ARE+=Sketch->CalculateARE(constData,actual_size[data]);
		fprintf_s(out,"Actual Size: %u Query Value: %u\n", a, query_val);
		// AAE+=Sketch->CalculateAAE_ML(constData,a,query_val);
		// Sketch->PrintCounterFile(constData,actual_size[data],counters);
	}
	AAE /= actual_size.size();
	// ARE /= actual_size.size();
	std::cout<<"AAE: "<<AAE<<std::endl;
	// std::cout<<"ARE: "<<ARE<<std::endl;
	fclose(out);
	fclose(counters);
	return 0;
}