#ifndef stm_allocator_h
#define stm_allocator_h

#include <unordered_map>

namespace stm
{
	struct BLK
	{
		void* data;
		unsigned int size;
	};

	class Allocator
	{
	public:
		void* allocate(unsigned int size)
		{
			void* data = _bucket[size];
			if (!data)
				data = malloc(size);
			else
				_bucket[size] = nullptr;
			return data;
		}

		void deallocate(unsigned int size, void* data)
		{
			auto& block = _bucket[size];
			if (block)
				free(data);
			else
				block = data;
		}

		~Allocator()
		{
			for (auto& it : _bucket)
				free(it.second);
		}

	private:
		std::unordered_map<unsigned int, void*> _bucket;
	};
}

#endif /* stm_allocator_h */
