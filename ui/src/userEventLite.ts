import { act, fireEvent } from '@testing-library/react';

/** Just enough of user-event for two tests: replace an input's value and click, each inside act(). */
export default {
  async type(el: HTMLElement, text: string) {
    await act(async () => { fireEvent.change(el, { target: { value: text } }); });
  },
  async click(el: HTMLElement) {
    await act(async () => { fireEvent.click(el); });
  },
};
