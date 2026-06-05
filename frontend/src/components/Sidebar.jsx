import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import ragLogo from '../assets/raglogo.png';

const Sidebar = () => {
  const location = useLocation();

  const stepLinks = [
    { to: '/load-file', text: '文档导入' },
    { to: '/chunk-file', text: '知识分块' },
    { to: '/parse-file', text: '文件解析' },
    { to: '/embedding', text: '向量存储' },
    { to: '/indexing', text: '向量库索引' },
    { to: '/search', text: '相似性检索' },
    { to: '/generation', text: '响应生成' },
  ];

  const isInSteps = stepLinks.some(l => l.to === location.pathname);
  const [stepsOpen, setStepsOpen] = useState(isInSteps);

  useEffect(() => {
    if (isInSteps) setStepsOpen(true);
  }, [location.pathname]);

  const linkClass = (path) =>
    `block px-4 py-3 text-gray-300 hover:bg-gray-700 transition-colors ${
      location.pathname === path ? 'bg-gray-700 border-l-2 border-blue-400' : ''
    }`;

  return (
    <div className="w-64 bg-gray-800 h-screen fixed left-0 top-0 flex flex-col">
      <div className="p-4 shrink-0">
        <img src={ragLogo} alt="Logo" className="w-full rounded" />
      </div>

      <nav className="flex-1 overflow-y-auto">
        {/* 步骤学习 collapsible folder */}
        <div>
          <button
            onClick={() => setStepsOpen(prev => !prev)}
            className={`w-full flex items-center justify-between px-4 py-3 text-gray-300 hover:bg-gray-700 transition-colors ${
              isInSteps && !stepsOpen ? 'bg-gray-700 border-l-2 border-blue-400' : ''
            }`}
          >
            <span className="flex items-center gap-2">
              <span className="text-gray-400">{stepsOpen ? '📂' : '📁'}</span>
              步骤学习
            </span>
            <span className="text-gray-500 text-xs">{stepsOpen ? '▲' : '▼'}</span>
          </button>

          {stepsOpen && (
            <div className="bg-gray-900">
              {stepLinks.map(link => (
                <Link
                  key={link.to}
                  to={link.to}
                  className={`block pl-10 pr-4 py-2 text-sm text-gray-400 hover:bg-gray-700 hover:text-gray-200 transition-colors ${
                    location.pathname === link.to
                      ? 'bg-gray-700 text-gray-200 border-l-2 border-blue-400'
                      : ''
                  }`}
                >
                  {link.text}
                </Link>
              ))}
            </div>
          )}
        </div>

        {/* 文件导入 */}
        <Link to="/import" className={linkClass('/import')}>
          <span className="flex items-center gap-2">
            <span className="text-gray-400">📥</span>
            文件导入
          </span>
        </Link>

        {/* 问答 */}
        <Link to="/qa" className={linkClass('/qa')}>
          <span className="flex items-center gap-2">
            <span className="text-gray-400">💬</span>
            问答
          </span>
        </Link>
      </nav>

      <div className="p-4 shrink-0">
        <p className="text-xs text-gray-600">在黄佳著的参考资料RAG框架项目上稍作修改</p>
      </div>
    </div>
  );
};

export default Sidebar;
