#include "TopK.h"
using namespace std;
TopK::TopK(uint type, uint d, uint w, uint k):k(k){
	sketch = new CMSketch(d, w);
	heap = new std::unordered_map<std::string, NO>();
	heap->clear();
	MP = heap->end(); // min pointer
	min = 1000000;    // flow count in heap
}

TopK::~TopK(){
	delete heap;
	sketch->~CMSketch();
}

// 外部注入訓練環境
// FILE* training_file;
// std::unordered_set<std::string>* actual_topk_set;
void TopK:: SetEnv(FILE* out, std::unordered_set<std::string>* label_set) {
	training_file = out;
	actual_topk_set = label_set;
	fprintf(training_file, "flow_id,initial_value,current_value,difference,c1,c2,c3,c4,label\n");
}

// 把 binary 字串轉成 hex 字串（方便當 ID）
std::string ToHex(const std::string& binary) {
	static const char* hex_chars = "0123456789ABCDEF";
	std::string hex;
	hex.reserve(binary.size() * 2); // 每 byte 2 字元
	for (unsigned char c : binary) {
		hex += hex_chars[c >> 4];
		hex += hex_chars[c & 0x0F];
	}
	return hex;
}

void TopK::Insert(cuc* str){
	sketch->Insert(str);
	std::string s = std::string((const char*)str);  // binary flow ID

	auto itr = heap->find(s);
	if (itr != heap->end()) {
		++itr->second;  // 更新 newValue
		if (itr == MP) {
			min++;
			for (itr = heap->begin(); itr != heap->end(); ++itr) {
				if (itr->second < min) {
					min = itr->second;
					MP = itr;
				}
			}
		}
		return;
	}
	//new flow
	// heap not full
	if (heap->size() != k) {
		uint initial = sketch->Query(str);
		heap->insert({s, NO(initial)});
		min = initial;
		MP = heap->find(s);
		return;
	}

	// heap full，檢查是否可擠掉
	uint v = sketch->Query(str);
	if (v > min) {
		// 擠掉 MP，並寫入 training record
		// std::string kicked_id = MP->first;
		// NO kicked = MP->second;
		// std::string hex_id = ToHex(kicked_id);
		// uint init = kicked.oldValue;
		// uint final = kicked.newValue;
		// int label = (actual_topk_set->count(kicked_id) > 0) ? 1 : 0;
		// fprintf(training_file, "%s,%u,%u,%u,%d\n", hex_id.c_str(), init, final, final - init, label);

		// 插入新流量
		heap->erase(MP);
		heap->insert({s, NO(v)});
		min = INT_MAX;
		for (auto it = heap->begin(); it != heap->end(); ++it) {
			if (it->second < min) {
				min = it->second;
				MP = it;
			}
		}
	}
}

void TopK::PrintTopK() {
	for (auto itr = heap->begin(); itr != heap->end(); ++itr) {
		std::cout << ToHex(itr->first) << " " << (uint)itr->second << std::endl;
	}
}

std::vector<std::pair<std::string, uint>> TopK::GetTopK() {
	std::vector<std::pair<std::string, uint>> topk_list;
	for (auto& it : *heap) {
		topk_list.emplace_back(it.first, (uint)it.second);
	}
	return topk_list;
}

// 將仍在 heap 裡的流量也輸出成 training data
void TopK::DumpToCSV(const std::string& filename) {
	FILE* fout = fopen(filename.c_str(), "w");
	if (!fout) {
		std::cerr << "Error opening output file: " << filename << std::endl;
		return;
	}

	fprintf(fout, "flow_id,initial_value,current_value,difference,c1,c2,c3,c4,label\n");

	for (auto& it : *heap) {
		std::string id = it.first;
		std::string hex_id = ToHex(id);
		uint init = it.second.oldValue;
		uint final = it.second.newValue;
		uint diff = final - init;

		// 原始 (未排序) counter 值
		std::vector<int> counters = sketch->GetCounter((cuc*)id.c_str());

		int label = (actual_topk_set->count(id) > 0) ? 1 : 0;

		fprintf(fout, "%s,%u,%u,%u", hex_id.c_str(), init, final, diff);
		for (int c : counters) {
			fprintf(fout, ",%d", c);
		}
		fprintf(fout, ",%d\n", label);
	}

	fclose(fout);
}