describe("KittenTTS React Native smoke", () => {
  it("generates speech metadata in the bare RN example", async () => {
    const input = await $("~tts-input");
    await input.waitForDisplayed({ timeout: 180000 });

    const generate = await $("~generate-button");
    await generate.waitForEnabled({ timeout: 180000 });
    await generate.click();

    const resultCard = await $("~result-card");
    await resultCard.waitForDisplayed({ timeout: 300000 });

    const sampleCount = await $("~sample-count");
    const duration = await $("~duration");

    const sampleCountValue = Number(
      (await sampleCount.getText()).replace(/[^0-9]/g, ""),
    );
    const durationValue = Number.parseFloat(
      (await duration.getText()).replace(/[^0-9.]/g, ""),
    );
    const sampleRateValue = Number(
      (await sampleRate.getText()).replace(/[^0-9]/g, ""),
    );
    const sampleHashValue = await sampleHash.getText();

    expect(sampleCountValue).toBeGreaterThan(0);
    expect(durationValue).toBeGreaterThan(0);
    expect(durationValue).toBeLessThan(30);
  });
});
