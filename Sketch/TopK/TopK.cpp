#include "TopK.h"

TopK::TopK(uint type, uint d, uint w, uint k):k(k){
	sketch = new CMSketch(d, w);
	heap = new std::unordered_map<std::string, NO>();
	heap->clear();
	MP = heap->end();//min pointer
	min = 1000000;//flow count in heap
}

TopK::~TopK(){
	delete heap;
}

void TopK::Insert(cuc* str){
	//insert to sketch
	sketch->Insert(str);

	//insert to heap
	std::unordered_map<std::string, NO>::iterator itr = heap->find(std::string((const char*)str));
	//if find the flow in heap
	if(itr!=heap->end()){
		++itr->second;
		//if is min pointer(MP)
		if(itr == MP) {
			min ++;
			//update min value
			for(itr = heap->begin(); itr != heap->end(); itr ++) {
				if(itr->second < min) {
					min = itr->second;
					MP = itr;
					break;
				}
			}
		}
		return;
	}
	//if flow not found in heap, and heap size <k then add into heap
	if(heap->size()!=k){ 
		std::string s = (const char*)str;
		heap->insert(std::make_pair(s, 1));
		min = 1;
		MP = heap->find(s);
		return;
	}
	//if sketch 值> min heap (min heap is full), 插入並更新minheap 
	uint v = sketch->Query(str);
	if(v > min) {
		heap->erase(MP);
		heap->insert(std::make_pair(std::string((const char*)str), v));
		std::unordered_map<std::string, NO>::iterator it;
		min = 1000000;
		for(it = heap->begin(); it != heap->end(); it ++) {
			if(it->second < min) {
				min = it->second;
				MP = it;
			}
		}
	}
	/*
	uint Min = INF_SAC;
	std::unordered_map<std::string, uint>::iterator top;
	for(itr = heap->begin(); itr != heap->end(); ++itr){
		if(Min > itr->second){
			Min = itr->second;
			top = itr;
		}
	}
	uint v = sketch->Query(str);
	if(v > Min){
		heap->erase(top);
		heap->insert(std::make_pair(std::string((const char*)str), v));
	}
	*/
}
// print top-k flows
void TopK::PrintTopK(){
	std::unordered_map<std::string, NO>::iterator itr;
	for(itr = heap->begin(); itr != heap->end(); ++itr){
		std::cout << itr->first << " " << itr->second << std::endl; 
	}
}
std::vector<std::pair<std::string, uint>> TopK::GetTopK() {
    std::vector<std::pair<std::string, uint>> topk_list;
    for (auto itr = heap->begin(); itr != heap->end(); ++itr) {
        topk_list.emplace_back(itr->first, (uint)itr->second);
    }
    return topk_list;
}
	