- float32: 32 bit = 4 bytes (1 bytes = 8 bit)

- Pytorch的切片、view、transpose操作是原地操作，返回的是内存地址而不是生成新的tensor

- transpose会造成内存不连续...这个也比较好理解，比如

  ```python
  x = torch.tensor([[0, 1, 2], [3, 4, 5]])
  y = x.transpose(1, 0)
  ```

  实际在内存中排列为`0, 1, 2, 3, 4, 5`，transpose后

  ```python
  y = [[0, 3], [1, 4], [2, 5]]
  # 此时内存中是跳着读取元素的，所以肯定不连续！！！
  y = y.contiguous()
  ```

  现在y的内存排序为`0, 3, 1, 4, 2, 5`，因此x与y不在共用内存地址。

- `reshape` = `contiguous().view()`

  ```python
  x = torch.tensor([[1, 2, 3], [4, 5, 6]])
  y = x.transpose(1, 0).reshape(2, 3)
  assert check_memory_address(x) != check_memory_address(y)
  ```

- tensor.rsqrt()

  ```python
  x = torch.tensor([1, 4, 9])
  assert torch.equal(t.rsqrt(), torch.tensor([1/1, 1/2, 1/3])) # x**(-0.5)，会生成新的内存地址
  ```

- 矩阵乘法计算量

  ```python
  x = torch.randn(2, 3)
  y = torch.randn(3, 2)
  x @ y
  '''
  输出shape为(2, 2)，因此总共有4次运算
  每次运算包含3个元素，由乘、加组成
  因此ops = 2 * 2 * 3 * 2
  '''
  ```

  <img src="1.jpg" alt="1" style="zoom: 33%;" />

