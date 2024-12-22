#include "CMSketch.h"

// #include "CUSketch.h"
// #include "CUSACSketch.h"
// #include "PSketch.h"
// #include "PSACSketch.h"
#include <fstream>
#define THRESHOLD 100
CMSketch *Sketch = new CMSketch(3, 9260);
std::unordered_map<std::string, ull> actual_size;

int main(){
	/*Insert your code here using the flowsize interface*/

	std::ifstream file("equinix-chicago1.dat",std::ios::binary);
	if(!file){
		std::cout<<"error, file not found!\n";
		return -1;
	}
	unsigned char* buffer = new unsigned char [13];
	memset(buffer,0,13);
	uint packet_count = 0;
	while(file.read(reinterpret_cast<char*>(buffer),13)|| file.gcount() > 0)
	{
		std::string data(reinterpret_cast<char*>(buffer), file.gcount());
		cuc* constData = buffer;
		Sketch->Insert(constData);
		actual_size[data]++;
		packet_count++;
	}
    // FILE* collisions;
    FILE* hash_values;
    hash_values = fopen("hash_values.txt","w");
    uint cid[92589][3];
    // collisions = fopen("collisions.txt","w+");
    int i = 0;
    float AAE = 0;
    float elephant_AAE = 0;
    for(auto it = actual_size.begin(); it != actual_size.end(); ++it) 
	{
        i++;
        cuc* temp = reinterpret_cast<cuc*>(const_cast<char*>(it->first.c_str()));
        uint second = it->second;
        Sketch->GetHashedValue(temp, cid[i]);
        fprintf(hash_values, "%u,%u,%u\n",cid[i][0],cid[i][1],cid[i][2]);
        if(second >= THRESHOLD)
            elephant_AAE+=Sketch->CalculateAAE(temp,second);
        AAE+=Sketch->CalculateAAE(temp,second);
    }
    float elephant_contribution = elephant_AAE / AAE;
    AAE/=actual_size.size();
    printf("AAE: %f, Elephant Flow Contribution: %.2f %%\n",AAE,elephant_contribution*100);
	return 0;
}