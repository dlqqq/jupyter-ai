import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import {
  ICompletionProviderManager,
  InlineCompletionTriggerKind,
  IInlineCompletionProvider,
  IInlineCompletionContext,
  IInlineCompletionList,
  IInlineCompletionItem,
  CompletionHandler
} from '@jupyterlab/completer';
import { ISettingRegistry } from '@jupyterlab/settingregistry';
import { Notification, showErrorMessage } from '@jupyterlab/apputils';
import { JSONValue, PromiseDelegate } from '@lumino/coreutils';
import {
  IEditorLanguageRegistry,
  IEditorLanguage
} from '@jupyterlab/codemirror';
import { NotebookPanel } from '@jupyterlab/notebook';
import { AiCompleterService as AiService } from './types';
import { DocumentWidget } from '@jupyterlab/docregistry';
import { jupyternautIcon } from '../icons';
import { getEditor } from '../selection-watcher';
import { IJupyternautStatus } from '../tokens';
import { CompletionWebsocketHandler } from './handler';

type StreamChunk = AiService.InlineCompletionStreamChunk;

/**
 * Format the language name nicely.
 */
export function displayName(language: IEditorLanguage): string {
  if (language.name === 'ipythongfm') {
    return 'Markdown (IPython)';
  }
  if (language.name === 'ipython') {
    return 'IPython';
  }
  return language.displayName ?? language.name;
}

export class JupyterAIInlineProvider implements IInlineCompletionProvider {
  readonly identifier = 'jupyter-ai';
  readonly icon = jupyternautIcon.bindprops({ width: 16, top: 1 });

  constructor(protected options: JupyterAIInlineProvider.IOptions) {
    options.completionHandler.modelChanged.connect(
      (_emitter, model: string) => {
        this._currentModel = model;
      }
    );
    options.completionHandler.streamed.connect(this._receiveStreamChunk, this);
  }

  get name(): string {
    if (this._currentModel.length > 0) {
      return `JupyterAI (${this._currentModel})`;
    } else {
      // This one is displayed in the settings.
      return 'JupyterAI';
    }
  }

  async fetch(
    request: CompletionHandler.IRequest,
    context: IInlineCompletionContext
  ): Promise<IInlineCompletionList<IInlineCompletionItem>> {
    const mime = request.mimeType ?? 'text/plain';
    const language = this.options.languageRegistry.findByMIME(mime);
    if (!language) {
      console.warn(
        `Could not recognise language for ${mime} - cannot complete`
      );
      return { items: [] };
    }
    if (!this.isLanguageEnabled(language?.name)) {
      // Do not offer suggestions if disabled.
      return { items: [] };
    }
    let cellId = undefined;
    let path = context.session?.path;
    if (context.widget instanceof NotebookPanel) {
      const activeCell = context.widget.content.activeCell;
      if (activeCell) {
        cellId = activeCell.model.id;
      }
    }
    if (!path && context.widget instanceof DocumentWidget) {
      path = context.widget.context.path;
    }
    const number = ++this._counter;

    const streamPreference = this._settings.streaming;
    const stream =
      streamPreference === 'always'
        ? true
        : streamPreference === 'never'
        ? false
        : context.triggerKind === InlineCompletionTriggerKind.Invoke;

    if (stream) {
      // Reset stream promises handler
      this._streamPromises.clear();
    }
    const result = await this.options.completionHandler.sendMessage({
      path: context.session?.path,
      mime,
      prefix: this._prefixFromRequest(request),
      suffix: this._suffixFromRequest(request),
      language: this._resolveLanguage(language),
      number,
      stream,
      cell_id: cellId
    });

    const error = result.error;
    if (error) {
      Notification.emit(`Inline completion failed: ${error.type}`, 'error', {
        autoClose: false,
        actions: [
          {
            label: 'Show Traceback',
            callback: () => {
              showErrorMessage('Inline completion failed on the server side', {
                message: error.traceback
              });
            }
          }
        ]
      });
      throw new Error(
        `Inline completion failed: ${error.type}\n${error.traceback}`
      );
    }
    return result.list;
  }

  /**
   * Stream a reply for completion identified by given `token`.
   */
  async *stream(token: string): AsyncGenerator<StreamChunk, void, unknown> {
    let done = false;
    while (!done) {
      const delegate = new PromiseDelegate<StreamChunk>();
      this._streamPromises.set(token, delegate);
      const promise = delegate.promise;
      yield promise;
      done = (await promise).done;
    }
  }

  get schema(): ISettingRegistry.IProperty {
    const knownLanguages = this.options.languageRegistry.getLanguages();
    return {
      properties: {
        maxPrefix: {
          title: 'Maximum prefix length',
          minimum: 1,
          type: 'number',
          description:
            'At most how many prefix characters should be provided to the model.'
        },
        maxSuffix: {
          title: 'Maximum suffix length',
          minimum: 0,
          type: 'number',
          description:
            'At most how many suffix characters should be provided to the model.'
        },
        disabledLanguages: {
          title: 'Disabled languages',
          type: 'array',
          items: {
            type: 'string',
            oneOf: knownLanguages.map(language => {
              return { const: language.name, title: displayName(language) };
            })
          },
          description:
            'Languages for which the completions should not be shown.'
        },
        streaming: {
          title: 'Streaming',
          type: 'string',
          oneOf: [
            { const: 'always', title: 'Always' },
            { const: 'manual', title: 'When invoked manually' },
            { const: 'never', title: 'Never' }
          ],
          description: 'Whether to show suggestions as they are generated'
        }
      },
      default: JupyterAIInlineProvider.DEFAULT_SETTINGS as any
    };
  }

  async configure(settings: { [property: string]: JSONValue }): Promise<void> {
    this._settings = settings as unknown as JupyterAIInlineProvider.ISettings;
  }

  isEnabled(): boolean {
    return this._settings.enabled;
  }

  isLanguageEnabled(language: string): boolean {
    return !this._settings.disabledLanguages.includes(language);
  }

  /**
   * Process the stream chunk to make it available in the awaiting generator.
   */
  private _receiveStreamChunk(
    _emitter: CompletionWebsocketHandler,
    chunk: StreamChunk
  ) {
    const token = chunk.response.token;
    if (!token) {
      throw Error('Stream chunks must return define `token` in `response`');
    }
    const delegate = this._streamPromises.get(token);
    if (!delegate) {
      console.warn('Unhandled stream chunk');
    } else {
      delegate.resolve(chunk);
      if (chunk.done) {
        this._streamPromises.delete(token);
      }
    }
  }

  /**
   * Extract prefix from request, accounting for context window limit.
   */
  private _prefixFromRequest(request: CompletionHandler.IRequest): string {
    const textBefore = request.text.slice(0, request.offset);
    const prefix = textBefore.slice(
      -Math.min(this._settings.maxPrefix, textBefore.length)
    );
    return prefix;
  }

  /**
   * Extract suffix from request, accounting for context window limit.
   */
  private _suffixFromRequest(request: CompletionHandler.IRequest): string {
    const textAfter = request.text.slice(request.offset);
    const prefix = textAfter.slice(
      0,
      Math.min(this._settings.maxPrefix, textAfter.length)
    );
    return prefix;
  }

  private _resolveLanguage(language: IEditorLanguage | null) {
    if (!language) {
      return 'plain English';
    }
    if (language.name === 'ipython') {
      return 'python';
    } else if (language.name === 'ipythongfm') {
      return 'markdown';
    }
    return language.name;
  }

  private _settings: JupyterAIInlineProvider.ISettings =
    JupyterAIInlineProvider.DEFAULT_SETTINGS;

  private _streamPromises: Map<string, PromiseDelegate<StreamChunk>> =
    new Map();
  private _currentModel = '';
  private _counter = 0;
}

export namespace JupyterAIInlineProvider {
  export interface IOptions {
    completionHandler: CompletionWebsocketHandler;
    languageRegistry: IEditorLanguageRegistry;
  }
  export interface ISettings {
    maxPrefix: number;
    maxSuffix: number;
    debouncerDelay: number;
    enabled: boolean;
    disabledLanguages: string[];
    streaming: 'always' | 'manual' | 'never';
  }
  export const DEFAULT_SETTINGS: ISettings = {
    maxPrefix: 10000,
    maxSuffix: 10000,
    // The debouncer delay handling is implemented upstream in JupyterLab;
    // here we just increase the default from 0, as compared to kernel history
    // the external AI models may have a token cost associated.
    debouncerDelay: 250,
    enabled: true,
    // ipythongfm means "IPython GitHub Flavoured Markdown"
    disabledLanguages: ['ipythongfm'],
    streaming: 'manual'
  };
}

export namespace CommandIDs {
  export const toggleCompletions = 'jupyter-ai:toggle-completions';
  export const toggleLanguageCompletions =
    'jupyter-ai:toggle-language-completions';
}

const INLINE_COMPLETER_PLUGIN =
  '@jupyterlab/completer-extension:inline-completer';

export const inlineCompletionProvider: JupyterFrontEndPlugin<void> = {
  id: 'jupyter_ai:inline-completions',
  autoStart: true,
  requires: [
    ICompletionProviderManager,
    IEditorLanguageRegistry,
    ISettingRegistry
  ],
  optional: [IJupyternautStatus],
  activate: async (
    app: JupyterFrontEnd,
    manager: ICompletionProviderManager,
    languageRegistry: IEditorLanguageRegistry,
    settingRegistry: ISettingRegistry,
    statusMenu: IJupyternautStatus | null
  ): Promise<void> => {
    if (typeof manager.registerInlineProvider === 'undefined') {
      // Gracefully short-circuit on JupyterLab 4.0 and Notebook 7.0
      console.warn(
        'Inline completions are only supported in JupyterLab 4.1+ and Jupyter Notebook 7.1+'
      );
      return;
    }
    const completionHandler = new CompletionWebsocketHandler();
    const provider = new JupyterAIInlineProvider({
      completionHandler,
      languageRegistry
    });
    await completionHandler.initialize();
    manager.registerInlineProvider(provider);

    const findCurrentLanguage = (): IEditorLanguage | null => {
      const widget = app.shell.currentWidget;
      const editor = getEditor(widget);
      if (!editor) {
        return null;
      }
      return languageRegistry.findByMIME(editor.model.mimeType);
    };

    let settings: ISettingRegistry.ISettings | null = null;

    settingRegistry.pluginChanged.connect(async (_emitter, plugin) => {
      if (plugin === INLINE_COMPLETER_PLUGIN) {
        // Only load the settings once the plugin settings were transformed
        settings = await settingRegistry.load(INLINE_COMPLETER_PLUGIN);
      }
    });

    app.commands.addCommand(CommandIDs.toggleCompletions, {
      execute: () => {
        if (!settings) {
          return;
        }
        const providers = Object.assign({}, settings.user.providers) as any;
        const ourSettings = {
          ...JupyterAIInlineProvider.DEFAULT_SETTINGS,
          ...providers[provider.identifier]
        };
        const wasEnabled = ourSettings['enabled'];
        providers[provider.identifier]['enabled'] = !wasEnabled;
        settings.set('providers', providers);
      },
      label: 'Enable Jupyternaut Completions',
      isToggled: () => {
        return provider.isEnabled();
      }
    });

    app.commands.addCommand(CommandIDs.toggleLanguageCompletions, {
      execute: () => {
        const language = findCurrentLanguage();
        if (!settings || !language) {
          return;
        }
        const providers = Object.assign({}, settings.user.providers) as any;
        const ourSettings = {
          ...JupyterAIInlineProvider.DEFAULT_SETTINGS,
          ...providers[provider.identifier]
        };
        const wasDisabled = ourSettings['disabledLanguages'].includes(
          language.name
        );
        const disabledList: string[] =
          providers[provider.identifier]['disabledLanguages'];
        if (wasDisabled) {
          disabledList.filter(name => name !== language.name);
        } else {
          disabledList.push(language.name);
        }
        settings.set('providers', providers);
      },
      label: () => {
        const language = findCurrentLanguage();
        return language
          ? `Enable Completions in ${displayName(language)}`
          : 'Enable Completions for Language of Current Editor';
      },
      isToggled: () => {
        const language = findCurrentLanguage();
        return !!language && provider.isLanguageEnabled(language.name);
      },
      isEnabled: () => {
        return !!findCurrentLanguage() && provider.isEnabled();
      }
    });

    if (statusMenu) {
      statusMenu.addItem({
        command: CommandIDs.toggleCompletions,
        rank: 1
      });
      statusMenu.addItem({
        command: CommandIDs.toggleLanguageCompletions,
        rank: 2
      });
    }
  }
};
