from PySide6.QtCore import QMutex, QThread


class YOLOThreadPool:
    MAX_THREADS = 3

    def __init__(self):
        self.threads_pool = {}  # 存放对象的字典
        self.thread_order = []  # 记录线程添加顺序的列表
        self._mutex = QMutex()  # 线程锁

    def _remove_oldest_thread(self):
        """移除最早添加的线程对象"""
        oldest_name = self.thread_order.pop(0)
        self.delete(oldest_name)

    def set(self, name, thread_obj):
        """设置或更新线程对象"""
        if not isinstance(thread_obj, QThread):
            raise ValueError("The object must be an instance of QThread.")

        self._mutex.lock()  # 加锁，确保线程安全
        # 如果已经存在，则停止并删除
        if name in self.threads_pool:
            self.thread_order.remove(name)
            self.delete(name)

        # 检查是否超过最大限制
        if len(self.threads_pool) >= self.MAX_THREADS:
            self._remove_oldest_thread()

        # 添加新线程
        self.threads_pool[name] = thread_obj
        self.thread_order.append(name)
        self._mutex.unlock()  # 解锁

    def get(self, name):
        """通过名称获取线程对象"""
        return self.threads_pool.get(name)

    def start_thread(self, name):
        """启动指定的线程对象"""
        thread_obj = self.get(name)
        if thread_obj and not thread_obj.isRunning():
            thread_obj.start()

    def stop_thread(self, name):
        """停止指定的线程对象"""
        thread_obj = self.get(name)
        if thread_obj and thread_obj.isRunning():
            thread_obj.quit()
            thread_obj.wait()

    def delete(self, name):
        """删除指定名称的线程对象"""
        thread = self.threads_pool.get(name)
        if thread and isinstance(thread, QThread):
            # 确保线程已停止
            if thread.isRunning():
                thread.quit()  # 请求线程退出
                thread.wait()  # 等待线程完全退出
            # 删除线程对象
            del self.threads_pool[name]


    def exists(self, name):
        """检查对象是否存在"""
        return name in self.threads_pool
